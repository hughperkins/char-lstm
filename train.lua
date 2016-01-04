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
require 'util/timer'
require 'util/file_helper'
require 'net'
require 'shared'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-dataset','tinyshakespeare','name of data directory. Should contain the file input.txt with input data')
cmd:option('-backend','cuda','cpu|cuda|cl')
cmd:text()

opt = cmd:parse(arg)

local backend = opt.backend
--local backend = 'cuda'
--backend = 'cl'
--backend = 'cpu'

if backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
elseif backend == 'cl' then
  require 'cltorch'
  require 'clnn'
end

--local dataset = 'tinyshakespeare'
local dataset = opt.dataset

local dataDir = 'data/' .. dataset

local dropoutProb = 0.0
local hiddenSizes = {128, 128}
local learningRate = 0.01
local seqLength = 50
local batchSize = 50
local dumpIntervalIts = 50

local outDir = 'out'

-- shamelessly adapted from Karpathy's char-rnn :-)
function text_to_t7(in_textfile, out_tensorfile, out_vocabfile)
  local cache_len = 10000
  local rawdata
  local tot_len = getFileSize(in_textfile)
  print('tot_len', tot_len)

  local input_coded = torch.ByteTensor(tot_len)
  local vocab = {}
  local ivocab = {}
  local state = {}
  state.vocab = vocab
  state.ivocab = ivocab

--  -- add a null character, which will be used for start-of-sequence
--  ivocab[1] = 0
--  vocab[0] = 1

  local f = io.open(in_textfile, "r")
  local input_string = f:read(cache_len)
  local pos = 1
  repeat
    local len = input_string:len()
    for i=1,len do
      local char = input_string:byte(i)
      if vocab[char] == nil then
        ivocab[#ivocab + 1] = char
        vocab[char] = #ivocab
      end
      local v = vocab[char]
      input_coded[pos] = v
      pos = pos + 1
    end
    input_string = f:read(cache_len)
  until not input_string
  f:close()

  print('#ivocab', #ivocab)
  local vocabs = {}
  vocabs.vocab = vocab
  vocabs.ivocab = ivocab
  torch.save(out_vocabfile, vocabs)
  torch.save(out_tensorfile, input_coded)
end

if not paths.dirp(outDir) then
  paths.mkdir(outDir)
end

if not paths.filep(dataDir .. '/' .. in_t7) or not paths.filep(dataDir .. '/' .. vocab_t7) then
  text_to_t7(dataDir .. '/' .. in_file, dataDir .. '/' .. in_t7, dataDir .. '/' .. vocab_t7)
--  local f = io.open(dataDir + '/' + in_file, 'r')
  
--  f:close()
end

local vocabs = torch.load(dataDir .. '/' .. vocab_t7)
local input = torch.load(dataDir .. '/' .. in_t7)
print('loaded input')
local ivocab = vocabs.ivocab
local vocab = vocabs.vocab

local netParams = {inputSize=#ivocab, hiddenSizes=hiddenSizes, dropout=dropoutProb}
local net, crit = makeNet(netParams)
print('net', net)
print('crit', crit)

net2 = nn.Sequencer(net)
crit2 = nn.SequencerCriterion(crit)

print('#ivocab', #ivocab)

local batchInput = torch.Tensor(batchSize, #ivocab)
if backend == 'cuda' then
  input = input:cuda()
  batchInput = batchInput:cuda()
  net:cuda()
  crit:cuda()
elseif backend == 'cl' then
  input = input:cl()
  batchInput = batchInput:cl()
  net:cl()
  crit:cl()
else
end

local input_len = input:size(1)
local batchSpacing = math.ceil(input_len / batchSize)
local inputDouble = input:clone():resize(2, input_len)
inputDouble[1]:copy(input)
inputDouble[2]:copy(input)
inputDouble = inputDouble:reshape(input_len * 2)
local inputStriped = inputDouble:unfold(1, input_len, batchSpacing):t()

local it = 1
local epoch = 1
--local it = 1
local itsPerEpoch = math.floor(input:size(1) / seqLength)
local params = net:getParameters()
net:training()
net:backwardOnline()
while true do
  sys.tic()
  local seqLoss = 0
  net:forget()
  net:zeroGradParameters()
  batchInputs = {}
  batchOutputs = {}

  local timer = timer_init()
  for s=1,seqLength do
    local batchOffset = ((epoch - 1) + (it - 1) * seqLength + (s - 1) + 1 - 1) % input_len + 1

    timer_update(timer, 'forward setup')
    local bc2 = inputStriped[batchOffset]
    batchInput:zero()
    batchInput:scatter(2, bc2:reshape(batchSize, 1), 1)

    timer_update(timer, 'forward run')
    os.exit()
    local batchOutput = net:forward(batchInput)
    batchInputs[s] = batchInput
    batchOutputs[s] = batchOutput
  end

  for s=seqLength,1,-1 do
    timer_update(timer, 'backward setup')
    local batchOffset = ((epoch - 1) + (it - 1) * seqLength + (s - 1) + 1 - 1) % input_len + 1
    local bt2 = inputStriped[batchOffset + 1]

    timer_update(timer, 'backward run')
    local batchLoss = crit:forward(batchOutputs[s], bt2)
    seqLoss = seqLoss + batchLoss
    local batchGradOutput = crit:backward(batchOutputs[s], bt2)
    net:backward(batchInputs[s], batchGradOutput)
  end
  timer_update(timer, 'end')
  timer_dump(timer)

  net:updateParameters(learningRate)
  print('epoch=' .. epoch, 'it=' .. it .. '/' .. itsPerEpoch, 'seqLoss=' .. seqLoss, 'time=' .. sys.toc())
  if it ~= 1 and ((it - 1) % dumpIntervalIts == 0) then
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

