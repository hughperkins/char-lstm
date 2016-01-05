--[[

This file trains a character-level multi-layer RNN on text data

Code is based on implementation in Andrej Karpathy's https://github.com/karpathy/char-rnn
... which was in turn based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on 
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)

]]--

require 'os'
require 'paths'
require 'torch'
require 'sys'
require 'nn'
require 'rnn'

-- local files
require 'util/timer'
require 'net'
require 'shared'
require 'data'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data','tinyshakespeare','name of data directory. Should contain the file input.txt with input data')
cmd:option('-back','cuda','cpu|cuda|cl')
cmd:option('-len', -1, 'only use this many characters from input data. -1 for all data')
cmd:option('-seq', 50, 'sequence length to use')
cmd:option('-dump', 100, 'iterations between writing out weights')
cmd:option('-hidden', '128,128', 'size of hidden layers, comma-separated, one per required hidden layer')
cmd:option('-drop',0 , 'dropout probability')
cmd:option('-batch',1 , 'batch size (should default to 50 probably)')
cmd:option('-lr',0.1, 'learning rate')
cmd:option('-profile', '', 'options file, written in lua')
cmd:option('-backprop','online', '(maintainers only) online|throughtime|noseq')
cmd:option('-maxepochs', -1, 'maximum epochs to train; -1 for no limit')
cmd:text()

opt = cmd:parse(arg)

if opt.profile ~= '' then
  print('profile', opt.profile)
  require(opt.profile)
end

local backend = opt.back

if backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
elseif backend == 'cl' then
  require 'cltorch'
  require 'clnn'
end

--local dataset = 'tinyshakespeare'
local dataset = opt.data

local dataDir = 'data/' .. dataset

local dropoutProb = opt.dropout
local hiddenSizes = opt.hidden:split(',')
local learningRate = opt.lr
local seqLength = opt.seq
local batchSize = opt.batch
local dumpIntervalIts = opt.dump
local max_input_length = opt.len

local outDir = 'out'

if not paths.dirp(outDir) then
  paths.mkdir(outDir)
end

local vocabs, ivocab, vocab, input = loadTextFile(dataDir) 

print('loaded input')

ivocab = truncateVocab(input, ivocab, max_input_length)

print('#ivocab', #ivocab)

local netParams = {inputSize=#ivocab, hiddenSizes=hiddenSizes, dropout=dropoutProb}
local net, crit = makeNet(netParams)
print('net', net)
print('crit', crit)

net2 = nn.Sequencer(net)
crit2 = nn.SequencerCriterion(crit)

local batchInputs = {}
local batchOutputs = {}
local batchOffsets = {}

for s=1,seqLength do
  batchInputs[s] = torch.FloatTensor(batchSize, #ivocab)
  batchOutputs[s] = torch.FloatTensor(batchSize, #ivocab)
end

if backend == 'cuda' then
  input = input:cuda()
  for s=1,seqLength do
    batchInputs[s] = batchInputs[s]:cuda()
    batchOutputs[s] = batchOutputs[s]:cuda()
  end
  net:cuda()
  crit:cuda()
elseif backend == 'cl' then
  input = input:cl()
  for s=1,seqLength do
    batchInputs[s] = batchInputs[s]:cl()
    batchOutputs[s] = batchOutputs[s]:cl()
  end
  net:cl()
  crit:cl()
else
  net:float()
  crit:float()
end

local input_len = input:size(1)
print('input_len', input_len)

local batchSpacing = math.ceil(input_len / batchSize)
local inputDouble = input.new():resize(2, input_len)
inputDouble[1]:copy(input)
inputDouble[2]:copy(input)
inputDouble:resize(input_len * 2)
local inputStriped = inputDouble:unfold(1, input_len, batchSpacing):t()
inputStriped:resize(input_len, batchSize)

function getLabel(vector)
  -- assumes a single 1-d vector of probabilities
  local max, label = vector:max(1)
  return label[1]
end

function populateBatchInput(batchOffset, debugState, inputStriped, batchInput)
  local bc2 = inputStriped[batchOffset]

  if debugState.printOutput then
    local thisCharCode = bc2[1]
    local thisChar = string.char(ivocab[thisCharCode])
    debugState.inputString = debugState.inputString .. thisChar
  end

  batchInput:zero()
  if backend == 'cpu' then
    batchInput:scatter(2, bc2:reshape(batchSize, 1):long(), 1)
  else
    batchInput:scatter(2, bc2:reshape(batchSize, 1), 1)
  end
end

-- what this does is, shift value up and down by an integer multiple
-- of modulus, so that it is in range 1 .. modulus
-- for example, if modules is 5, then for the following input values
-- we will get the following output values
-- input  output
-- 0      5
-- 1      1
-- 2      2
-- 3      3
-- 4      4
-- 5      5
-- 6      1
-- 7      2
-- (this is different from normal lua '%' function, which will put
-- into range 0 .. (modulus-1), which is quite not lua-like :-)
function lua_modulus(value, modulus)
  return (value - 1) % modulus + 1
end

function doOutputDebug(debugState, batchOutput)
  if debugState.printOutput then
--    print('batchInput', batchInput:narrow(2,1,seqLength):reshape(1,seqLength))
    print('batchOutput:exp()', batchOutput:exp():narrow(2,1,debugState.seqLength):reshape(1, debugState.seqLength))
    local label = getLabel(batchOutput[1])
    debugState.outputString = debugState.outputString .. string.char(debugState.ivocab[label])
  end
end

function doBackwardDebug(debugState, batchTarget, batchLoss, batchGradOutput)
  if debugState.printOutput then
--    print('batchTarget', batchTarget)
    local thisCharCode = batchTarget[1]
    local thisChar = string.char(ivocab[thisCharCode])
    debugState.targetString = thisChar .. debugState.targetString
    print('batchLoss', batchLoss)
--    print('batchGradOutput', batchGradOutput)
  end
end

function makeBatchTarget(targetOffset, debugState, inputStriped)
  local batchTarget = inputStriped[targetOffset]
  return batchTarget
end

local it = 1
local epoch = 1
--local it = 1
local itsPerEpoch = math.floor(input:size(1) / seqLength)
if itsPerEpoch < 1 then
  error('seqlength cannot be smaller than input size')
end
local params = net:getParameters()
net:training()
while opt.maxepochs <= 0 or epoch <= opt.maxepochs do
  sys.tic()
  local seqLoss = 0

  local timer = timer_init()
  local debugState = {}
  debugState.printOutput = false
  debugState.inputString = ''
  debugState.targetString = ''
  debugState.outputString = ''
  debugState.seqLength = seqLength
  debugState.ivocab = ivocab
  local globalIt = epoch * itsPerEpoch + it
--  print('epoch', epoch, 'itsPerEpoch', itsPerEpoch, 'it', it, it % 10)
  if globalIt % 50 == 0 then
--  if true then
    print('======================')
    print('globalIt', globalIt, 'epoch', epoch, 'it', it)
    debugState.printOutput = true
  end
  local epochOffset = epoch - 1
  epochOffset = 0
  if opt.backprop == 'online' then

    net:forget()
    net:zeroGradParameters()
    net:backwardOnline()
    for s=1,seqLength do
      batchOffsets[s] = lua_modulus(epochOffset + (it - 1) * seqLength + (s - 1) + 1, input_len)
      timer_update(timer, 'forward setup')
      populateBatchInput(batchOffsets[s], debugState, inputStriped, batchInputs[s])

      timer_update(timer, 'forward run')
      local batchOutput = net:forward(batchInputs[s])
--      batchInputs[s] = batchInput
      batchOutputs[s]:copy(batchOutput)

--      doOutputDebug(debugState, batchOutput)
    end

    for s=seqLength,1,-1 do
      timer_update(timer, 'backward setup')
      local targetOffset = lua_modulus(batchOffsets[s] + 1, input_len)
      local batchTarget = makeBatchTarget(targetOffset, debugState, inputStriped)

      timer_update(timer, 'backward run')
      local batchLoss = crit:forward(batchOutputs[s], batchTarget)
      local batchGradOutput = crit:backward(batchOutputs[s], batchTarget)
      net:backward(batchInputs[s], batchGradOutput)

--      if debugState.printOutput then
--        print('batchInputs[s]', batchInputs[s])
--        print('batchTarget', batchTarget)
--        print('batchGradOutput', batchGradOutput)
--      end

      seqLoss = seqLoss + batchLoss
--      doBackwardDebug(debugState, batchTarget, batchLoss, batchGradOutput)
    end
    net:updateParameters(learningRate)
  elseif opt.backprop == 'throughtime' then
    if #hiddenSizes > 1 then
      error('cannot use backwardsthroughtime with more than one hidden layer, should use backwardsonline instead')
    end

    net:forget()
    net:zeroGradParameters()
--    net:backwardOnline()
    for s=1,seqLength do
      batchOffsets[s] = lua_modulus(epochOffset + (it - 1) * seqLength + (s - 1) + 1, input_len)

      timer_update(timer, 'forward setup')
      populateBatchInput(batchOffsets[s], debugState, inputStriped, batchInputs[s])

      timer_update(timer, 'forward run')
      local batchOutput = net:forward(batchInputs[s])
--      batchInputs[s] = batchInput
      batchOutputs[s]:copy(batchOutput)

--      doOutputDebug(debugState, batchOutput)
--    end

--    for s=seqLength,1,-1 do
      timer_update(timer, 'backward setup')
      local targetOffset = lua_modulus(batchOffsets[s] + 1, input_len)
      local batchTarget = makeBatchTarget(targetOffset, debugState, inputStriped)

      timer_update(timer, 'backward run')
      local batchLoss = crit:forward(batchOutputs[s], batchTarget)
      local batchGradOutput = crit:backward(batchOutputs[s], batchTarget)
      net:backward(batchInputs[s], batchGradOutput)

--      if debugState.printOutput then
--        print('batchInputs[s]', batchInputs[s])
--        print('batchTarget', batchTarget)
--        print('batchGradOutput', batchGradOutput)
--      end

      seqLoss = seqLoss + batchLoss
--      doBackwardDebug(debugState, batchTarget, batchLoss, batchGradOutput)
    end
    net:backwardThroughTime()
    net:updateParameters(learningRate)
  elseif opt.backprop == 'noseq' then
    for s=1,seqLength do
      net:forget()
      net:zeroGradParameters()
      local batchOffset = lua_modulus(epochOffset + (it - 1) * seqLength + (s - 1) + 1, input_len)
      local targetOffset = lua_modulus(batchOffset + 1, input_len)

      timer_update(timer, 'forward setup')
      populateBatchInput(batchOffset, debugState, inputStriped, batchInputs[s])

      timer_update(timer, 'backward setup')
      local batchTarget = makeBatchTarget(targetOffset, debugState, inputStriped)

      timer_update(timer, 'forward run')
      local batchOutput = net:forward(batchInputs[s])
--      doOutputDebug(debugState, batchOutput)

      timer_update(timer, 'backward run')
      local batchLoss = crit:forward(batchOutput, batchTarget)
      local batchGradOutput = crit:backward(batchOutput, batchTarget)
      net:backward(batchInputs[s], batchGradOutput)
      if debugState.printOutput then
        
      end
      net:updateParameters(learningRate)

      seqLoss = seqLoss + batchLoss
--      doBackwardDebug(debugState, batchTarget, batchLoss, batchGradOutput)
    end
  else
    error('invalid opt.backprop value')
  end
  timer_update(timer, 'end')

  if debugState.printOutput then
--    timer_dump(timer)
--    print('params:narrow(1,1,seqLength)', params:narrow(1,1,seqLength):reshape(1,seqLength))
--    print('input ', debugState.inputString)
--    print('target', debugState.targetString)
--    print('output', debugState.outputString)
    print('epoch=' .. epoch, 'it=' .. it .. '/' .. itsPerEpoch, 'seqLoss=' .. seqLoss, 'time=' .. sys.toc())
  end
  if globalIt ~= 1 and ((globalIt - 1) % dumpIntervalIts == 0) then
    local filename = weights_t7:gsub('$DATASET', dataset):gsub('$EPOCH', epoch):gsub('$IT', it)
    print('filename', filename)
    local data = {}
    data.dataset = dataset
    data.netParams = netParams
    data.weights = params:float()
    data.backend = backend
    torch.save(outDir .. '/' .. filename, data) 
  end
  it = it + 1
  if it > itsPerEpoch then
    epoch = epoch + 1
    it = 1
  end
end

