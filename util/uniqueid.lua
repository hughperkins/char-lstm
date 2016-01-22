function os.hostname()
   local f = io.popen ("/bin/hostname")
   if not f then 
      return 'localhost'
   end
   local hostname = f:read("*a") or ""
   f:close()
   hostname =string.gsub(hostname, "\n$", "")
   return hostname
end

-- Generates a globally unique identifier.
-- If a namespace is provided it is concatenated with 
-- the time of the call, and the next value from a sequence
-- to get a pseudo-globally-unique name.
-- Otherwise, we concatenate the linux hostname
local counter = 1
function os.uniqueid(namespace, separator)
   local separator = separator or ':'
   local namespace = namespace or os.hostname()
   local uid = namespace..separator..os.time()..separator..counter
   counter = counter + 1
   return uid
end
