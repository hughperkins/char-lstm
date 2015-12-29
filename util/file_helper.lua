require 'sys'

function getFileSize(filePath)
  local sizestr = sys.execute('stat --format=%s ' .. filePath)
  local size = tonumber(sizestr)
  return size
end


