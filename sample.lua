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

local data_file = arg[1]
print('data_file', data_file)

local data = torch.load(data_file)

local net, crit = makeNet(data.netParams)
local params = net:getParameters()
params:copy(data.weights)

local backend = data.backend
if backend == 'cuda' then
  require 'cunn'
  net:cuda()
elseif backend == 'cl' then
  require 'clnn'
  net:cl()
else
end

print('net', net)
params = net:getParameters()
print('torch.type(params)', torch.type(params))

--local netandcrit = torch.load('out/critandnet.t7')

