
require 'os'
require 'paths'
require 'torch'
require 'sys'
require 'shared'
local lfs = require "lfs"

-- shamelessly adapted from Karpathy's char-rnn :-)
function text_to_t7(in_textfile, out_tensorfile, out_vocabfile)
  local cache_len = 10000
  local rawdata
  local tot_len = lfs.attributes (in_textfile, "size")
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

function loadTextFile(dataDir)
  if not paths.filep(dataDir .. '/' .. in_t7) or not paths.filep(dataDir .. '/' .. vocab_t7) then
    text_to_t7(dataDir .. '/' .. in_file, dataDir .. '/' .. in_t7, dataDir .. '/' .. vocab_t7)
  end

  local vocabs = torch.load(dataDir .. '/' .. vocab_t7)
  local ivocab = vocabs.ivocab
  local vocab = vocabs.vocab
  local input = torch.load(dataDir .. '/' .. in_t7)
  
  return vocabs, ivocab, vocab, input
end

function truncateVocab(input, ivocab, max_input_length)
  if max_input_length > 0 then
    print('max_input_length', max_input_length)
    input:resize(max_input_length)
    if max_input_length < #ivocab then
      local newivocab = {}
      for i=1,max_input_length do
        newivocab[i] = ivocab[i]
      end
      ivocab = newivocab
      print('#ivocab', #ivocab)
    end
  end
  return ivocab
end
