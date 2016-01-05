require 'sys'
local lfs = require "lfs"

function getFileSize(filePath)
  local size = lfs.attributes (filePath, "size")
  return size
end


