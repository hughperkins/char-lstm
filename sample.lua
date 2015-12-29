--[[

This file samples characters from a trained model

Code is based on implementation in Andrej Karpathy's https://github.com/karpathy/char-rnn
which was in turn based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6

]]--

require 'nn'
require 'rnn'
require 'sys'
require 'os'
require 'torch'
require 'paths'
require 'shared'
require 'net'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
cmd:argument('-model','model checkpoint to use for sampling')
cmd:option('-backend', 'cuda', 'cpu|cuda|cl')
cmd:option('-length', 2000, 'number of characters to sample')
cmd:option('-temperature', 1, 'temperature of sampling')
cmd:text()

opt = cmd:parse(arg)

local data_file = opt.model

local data = torch.load(data_file)
local dataset = data.dataset
local dataDir = 'data/' .. dataset

local net, crit = makeNet(data.netParams)
local params = net:getParameters()
params:copy(data.weights)

print('dataDir', dataDir)
local vocabs = torch.load(dataDir .. '/' .. vocab_t7)
local input = torch.load(dataDir .. '/' .. in_t7)
print('loaded input')
local ivocab = vocabs.ivocab
local vocab = vocabs.vocab

local identity = torch.eye(#ivocab)

--local backend = data.backend
local backend = opt.backend
print('backend', backend)
if backend == 'cuda' then
  require 'cunn'
  net:cuda()
  identity = identity:cuda()
elseif backend == 'cl' then
  require 'clnn'
  net:cl()
  identity = identity:cl()
else
end

print('net', net)
params = net:getParameters()
print('torch.type(params)', torch.type(params))

function sampleSoftMax(log_q, t)
  -- eg see https://en.wikipedia.org/wiki/Softmax_function, section 'Reinforcement learning'
  local q = log_q:clone():exp()
  q:mul(1/t)
--  print('q:sum()', q:sum())
  local sum_q = q:sum()
  q:div(sum_q)
--  print('q:sum()', q:sum())
  local sample = torch.multinomial(q, 1)
  return sample
end

--local netandcrit = torch.load('out/critandnet.t7')
-- seed with '\n' for now. a bit too much prior knowledge introduced by doing this, but
-- gets it working for now....
local newLine = '\n'
--print('vocab', vocab)
--for k,v in pairs(vocab) do
--  print('vocab', k, v)
--end
--for k,v in pairs(ivocab) do
--  print('ivocab', k, v, torch.type(k), torch.type(v))
--end
local newLineCode = newLine:byte(1)
--print('newLineCode', newLineCode)
local prevChar = vocab[newLineCode]
net:evaluate()
local sample = ''
for i=1,opt.length do
  local output = net:forward(identity[prevChar])
  local outputexp = output:clone():exp()
--  print('outputexp', outputexp)
--  print('outputexp:sum()', outputexp:sum())
  local thisChar = sampleSoftMax(output, opt.temperature)
--  print('thisChar', thisChar, 'torch.type(thisChar[1])', torch.type(thisChar[1]))
--  print('thisChar', thisChar)
--  print('ivocab[thisChar]', ivocab[thisChar[1]])
  sample = sample .. string.char(ivocab[thisChar[1]])
  prevChar = thisChar[1]
end
print(sample)

