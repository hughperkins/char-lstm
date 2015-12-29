require 'sys'

function timer_init()
  local state = {}
  return state
--  state.time = sys.clock()
end

function timer_update(state, name)
  local now = sys.clock()
  if state.time ~= nil then
    local thisInterval = now - state.time
    state[state.name] = state[state.name] or 0
    state[state.name] = state[state.name] + thisInterval
  end
  state.time = now
  state.name = name
  return state
end

function timer_dump(state)
  for k,v in pairs(state) do
    if k ~= 'time' and k ~= 'name' then
      print(k .. '=' .. v)
    end
  end
end


