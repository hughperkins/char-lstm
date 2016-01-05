require 'nn'
require 'rnn'

function makeNet(params)
  local inputSize = params.inputSize
  local hiddenSizes = params.hiddenSizes
  local dropout = params.dropout

  local net = nn.Sequential()
  local thisInputSize = inputSize
  print('hiddenSizes', hiddenSizes)
  for i, hiddenSize in ipairs(hiddenSizes) do
    net:add(nn.FastLSTM(thisInputSize, hiddenSize))
    if dropout ~= nil and dropout > 0 then
      net:add(nn.Dropout(dropout))
    end
    thisInputSize = hiddenSize
  end

  net:add(nn.Linear(thisInputSize, inputSize))
  net:add(nn.LogSoftMax())

  local recursor = nn.Recursor(net)

  local crit = nn.ClassNLLCriterion()
--  return net, crit
  return recursor, crit
end

