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
require 'dpnn'
require 'rnn'

-- local files
require 'util/timer'
require 'util/uniqueid'
require 'net'
require 'shared'
require 'data'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data', 'tinyshakespeare', 'name of data directory. Should contain the file input.txt with input data')
cmd:option('-back', 'cuda', 'cpu|cuda|cl')
cmd:option('-seq', 50, 'sequence length to use')
cmd:option('-hidden', '128,128', 'size of hidden layers, comma-separated, one per required hidden layer')
cmd:option('-drop', 0 , 'dropout probability')
cmd:option('-batchsize', 50 , 'batch size')
cmd:option('-lr', 0.1, 'learning rate')
cmd:option('-mom', 0.9, 'momentum')
cmd:option('-clip', 5, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set. test_frac will be computed as (1 - train_frac - val_frac)')
cmd:option('-profile', '', 'options file, written in lua')
cmd:option('-epochsize', -1, 'every how many iterations (batches) should we evaluate on validation data?')
cmd:option('-maxepoch', -1, 'maximum epochs to train; -1 for no limit')
cmd:option('-earlystop', 50, 'early-stop when the model cant find a new validation NLL minima for this many epochs')
cmd:option('-outdir', 'out', 'name of output directory')
cmd:option('-progress', false, 'print training progress bar')
cmd:text()

local opt = cmd:parse(arg)
opt.hiddenSizes = opt.hidden:split(',')

if opt.profile ~= '' then
  print('profile', opt.profile)
  opt = require(opt.profile)
end

if opt.back == 'cuda' then
  require 'cutorch'
  require 'cunn'
elseif opt.back == 'cl' then
  require 'cltorch'
  require 'clnn'
end

if not paths.dirp(opt.outdir) then
  paths.mkdir(opt.outdir)
end

-- load data from file
local dataDir = 'data/' .. opt.data
local splitFrac = {opt.train_frac, opt.val_frac, nil}
local loader = nn.CharTextLoader(dataDir, opt.batchsize, opt.seq, splitFrac)
local vocab_size = loader.vocab_size  -- the number of distinct characters
print('#vocab', loader.vocab_size)
print('nChar (total)', loader.text_size)
print('nBatch (train/val/test)', loader.ntrain, loader.nval, loader.ntest)

-- build model
local net, crit = makeLSTM(loader.vocab_size, opt.hiddenSizes, opt.dropout)
print('net', net)
print('crit', crit)

opt.id = opt.data .. ':' .. os.uniqueid()

print("options")
print(opt)

if backend == 'cuda' then
  net:cuda(); crit:cuda()
elseif backend == 'cl' then
  net:cl(); crit:cl()
else
  net:float(); crit:float()
end

-- experiment log.
-- will be saved to file every time a new validation minima is found
local xplog = {}
xplog.opt = opt -- save all hyper-parameters and such
xplog.vocabSize = loader.vocab_size
-- will only serialize params
xplog.model = nn.Serial(net)
xplog.model:mediumSerial(false)
-- keep a log of NLL for each epoch
xplog.trainNLL = {}
xplog.valNLL = {}
-- will be used for early-stopping
xplog.minValNLL = 99999999
xplog.epoch = 0
local ntrial = 0

local epoch = 1
opt.epochsize = opt.epochsize == -1 and loader.ntrain or opt.epochsize

while opt.maxepoch <= 0 or epoch <= opt.maxepoch do
  print("")
  print("Epoch #"..epoch.." :")
  
  -- 1. training
  
  local sumErr = 0
  net:training()
  for i=1,opt.epochsize do
    local timer = timer_init()
    
    -- get batch of sequences
    -- input and target are each tensors of size seqLen x batchSize
    local input, target = loader:next_batch(loader.TRAIN, true)

    -- forward
    timer_update(timer, 'forward')
    local output = net:forward(input)
    local err = crit:forward(output, target)
    sumErr = sumErr + err
    
    -- backward
    timer_update(timer, 'backward')
    local gradOutput = crit:backward(output, target)
    net:zeroGradParameters()
    net:backward(input, gradOutput)
    
    -- update
    timer_update(timer, 'update')
    if opt.clip > 0 then
      local norm = net:gradParamClip(opt.clip) -- affects gradParams
      opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
    end
    --net:updateGradParameters(opt.mom) -- momentum : affects gradParams
    net:updateParameters(opt.lr) -- affects params    
    
    if opt.progress then
      xlua.progress(i, opt.epochsize)
    end
    
    timer_update(timer, 'end')
  end
  
  local nll = sumErr/(opt.epochsize*opt.seq)
  print("Training NLL : "..nll)
  
  xplog.trainNLL[epoch] = nll
  
  -- 2. evaluation
  
  local sumErr = 0
  for i=1,loader.nval do
    local input, target = loader:next_batch(loader.VAL, true)

    local output = net:forward(input)
    local err = crit:forward(output, target)
    sumErr = sumErr + err
  end
  
  local nll = sumErr/(loader.nval*opt.seq)
  print("Validation NLL : "..nll)
  
  xplog.valNLL[epoch] = nll
  ntrial = ntrial + 1
  
  -- early-stopping
  if nll < xplog.minValNLL then
    -- save best version of model
    xplog.minValNLL = nll
    xplog.epoch = epoch 
    local filename = path.join(opt.outdir, opt.id..'.t7')
    print("Found new minima. Saving to : "..filename)
    torch.save(filename, xplog)
    ntrial = 0
  elseif ntrial >= opt.earlystop then
    print("No new minima found after "..ntrial.." epochs.")
    print("Stopping experiment.")
    os.exit()
  end

  epoch = epoch + 1
end

