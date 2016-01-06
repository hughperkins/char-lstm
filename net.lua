require 'nn'
require 'rnn'

function makeLSTM(inputSize, hiddenSizes, dropout, gpu)
  assert(nn.OneHot, "Please update dpnn : luarocks install dpnn")

  -- build core stack of LSTM + output layer.
  -- lstm will be applied to each step in sequence
  local lstm = nn.Sequential()
  local thisInputSize = inputSize
  for i, hiddenSize in ipairs(hiddenSizes) do
    local hiddenSize = tonumber(hiddenSize)
    lstm:add(nn.FastLSTM(thisInputSize, hiddenSize))
    if dropout ~= nil and dropout > 0 then
      lstm:add(nn.Dropout(dropout))
    end
    thisInputSize = hiddenSize
  end

  lstm:add(nn.Linear(thisInputSize, inputSize))
  lstm:add(nn.LogSoftMax())
  
  -- input to model is seqLen x batchSize
  
  local model = nn.Sequential()
  model:add(nn.Convert()) -- casts input to type of model (dpnn).
  model:add(nn.OneHot(inputSize)) -- converts indices to onehot (dpnn).
  model:add(nn.SplitTable(1, 3)) -- splits sequence tensor into a table of tensors
  
  local seq = nn.Sequencer(lstm)
  -- remember hidden states between batches for both training and evaluation
  seq:remember('both') 
  model:add(seq)
  
  -- build criterion

  -- target is also seqLen x batchSize.
  local targetmodule = nn.SplitTable(1, 2)
  if gpu then
    targetmodule =nn.Sequential()
    :add(nn.Convert())
    :add(targetmodule)
  end
    
  local crit = nn.ModuleCriterion(
    nn.SequencerCriterion(nn.ClassNLLCriterion()), 
    nil, 
    targetmodule
  )
  
  return model, crit
end

