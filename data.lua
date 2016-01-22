require 'os'
require 'paths'
require 'torch'
require 'sys'
local lfs = require "lfs"

-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits
-- Modified from https://github.com/karpathy/char-rnn  (MIT License)
-- modifications include name change and use of torch.class (not many)

local CharTextLoader = torch.class("nn.CharTextLoader")

CharTextLoader.TRAIN = 1
CharTextLoader.VAL = 2
CharTextLoader.TEST = 3

function CharTextLoader:__init(data_dir, batch_size, seq_length, split_fractions)
  -- split_fractions is e.g. {0.9, 0.05, 0.05} (train, valid, test)
  
  -- train / val / test split for data, in fractions
  split_fractions[3] = split_fractions[3] or math.max(0, 1 - (split_fractions[1] + split_fractions[2]))

  local input_file = paths.concat(data_dir, 'input.txt')
  local vocab_file = paths.concat(data_dir, 'vocab.t7')
  local tensor_file = paths.concat(data_dir, 'data.t7')

  -- fetch file attributes to determine if we need to rerun preprocessing
  local run_prepro = false
  if not (paths.filep(vocab_file) or paths.filep(tensor_file)) then
    -- prepro files do not exist, generate them
    print('vocab.t7 and data.t7 do not exist. Running preprocessing...')
    run_prepro = true
  else
    -- check if the input file was modified since last time we 
    -- ran the prepro. if so, we have to rerun the preprocessing
    local input_attr = lfs.attributes(input_file)
    local vocab_attr = lfs.attributes(vocab_file)
    local tensor_attr = lfs.attributes(tensor_file)
    if input_attr.modification > vocab_attr.modification or input_attr.modification > tensor_attr.modification then
      print('vocab.t7 or data.t7 detected as stale. Re-running preprocessing...')
      run_prepro = true
    end
  end
  if run_prepro then
    -- construct a tensor with all the data, and vocab file
    print('one-time setup: preprocessing input text file ' .. input_file .. '...')
    self.text_to_tensor(input_file, vocab_file, tensor_file)
  end

  print('loading data files...')
  local data = torch.load(tensor_file)
  self.vocab = torch.load(vocab_file)

  -- cut off the end so that it divides evenly
  local len = data:size(1)
  if len % (batch_size * seq_length) ~= 0 then
    print('cutting off end of data so that the batches/sequences divide evenly')
    data = data:sub(1, batch_size * seq_length * math.floor(len / (batch_size * seq_length)))
  end
  self.text_size = data:size(1)

  -- count vocab
  self.vocab_size = 0
  for _ in pairs(self.vocab) do 
    self.vocab_size = self.vocab_size + 1 
  end

  -- self.batches is a table of tensors
  print('reshaping tensor...')
  self.batch_size = batch_size
  self.seq_length = seq_length

  local ydata = data:clone()
  ydata:sub(1,-2):copy(data:sub(2,-1))
  ydata[-1] = data[1]
  self.x_batches = data:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
  self.nbatches = #self.x_batches
  self.y_batches = ydata:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
  assert(#self.x_batches == #self.y_batches)

  -- lets try to be helpful here
  if self.nbatches < 50 then
    print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
  end

  -- perform safety checks on split_fractions
  assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
  assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
  assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')
  if split_fractions[3] == 0 then 
    -- catch a common special case where the user might not want a test set
    self.ntrain = math.floor(self.nbatches * split_fractions[1])
    self.nval = self.nbatches - self.ntrain
    self.ntest = 0
  else
    -- divide data to train/val and allocate rest to test
    self.ntrain = math.floor(self.nbatches * split_fractions[1])
    self.nval = math.floor(self.nbatches * split_fractions[2])
    self.ntest = self.nbatches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)
    assert(self.ntest >= 0)
  end

  self.split_sizes = {self.ntrain, self.nval, self.ntest}
  self.batch_ix = {0,0,0}

  print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))
  collectgarbage()
  return self
end

function CharTextLoader:reset_batch_pointer(split_index, batch_index)
  batch_index = batch_index or 0
  self.batch_ix[split_index] = batch_index
end

-- returned input and target are each tensors of size : batchSize x seqLen
function CharTextLoader:next_batch(split_index, seqFirst)
  if self.split_sizes[split_index] == 0 then
    -- perform a check here to make sure the user isn't screwing something up
    local split_names = {'train', 'val', 'test'}
    error('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
  end
  -- split_index is integer: 1 = train, 2 = val, 3 = test
  self.batch_ix[split_index] = self.batch_ix[split_index] + 1
  if self.batch_ix[split_index] > self.split_sizes[split_index] then
    self.batch_ix[split_index] = 1 -- cycle around to beginning
  end
  -- pull out the correct next batch
  local ix = self.batch_ix[split_index]
  if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
  if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val
  local input, target = self.x_batches[ix], self.y_batches[ix]
  if seqFirst then
    input, target = input:t(), target:t()
  end
  return input, target
end

-- *** STATIC method ***
function CharTextLoader.text_to_tensor(in_textfile, out_vocabfile, out_tensorfile)
  local timer = torch.Timer()

  print('loading text file...')
  local cache_len = 10000
  local rawdata
  local tot_len = 0
  local f = assert(io.open(in_textfile, "r"))

  -- create vocabulary if it doesn't exist yet
  print('creating vocabulary mapping...')
  
  -- record all characters to a set
  local unordered = {}
  rawdata = f:read(cache_len)
  repeat
    for char in rawdata:gmatch'.' do
      if not unordered[char] then 
        unordered[char] = true 
      end
    end
    tot_len = tot_len + #rawdata
    rawdata = f:read(cache_len)
  until not rawdata
  f:close()
  
  -- sort into a table (i.e. keys become 1..N)
  local ordered = {}
  for char in pairs(unordered) do 
    ordered[#ordered + 1] = char 
  end
  table.sort(ordered)
  
  -- invert `ordered` to create the char->int mapping
  local vocab = {}
  for i, char in ipairs(ordered) do
      vocab[char] = i
  end
  
  -- construct a tensor with all the data
  print('putting data into tensor...')
  local data = torch.ByteTensor(tot_len) -- store it into 1D first, then rearrange
  f = assert(io.open(in_textfile, "r"))
  local currlen = 0
  rawdata = f:read(cache_len)
  repeat
    for i=1, #rawdata do
      data[currlen+i] = vocab[rawdata:sub(i, i)] -- lua has no string indexing using []
    end
    currlen = currlen + #rawdata
    rawdata = f:read(cache_len)
  until not rawdata
  f:close()

  -- save output preprocessed files
  print('saving ' .. out_vocabfile)
  torch.save(out_vocabfile, vocab)
  print('saving ' .. out_tensorfile)
  torch.save(out_tensorfile, data)
  
  return vocab, data
end
