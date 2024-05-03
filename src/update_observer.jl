function update_observer!(observer; kwargs...)
  return error("Not implemented")
end

using ITensors.ITensorMPS: ITensorMPS
function update_observer!(observer::ITensorMPS.AbstractObserver; kwargs...)
  return ITensorMPS.measure!(observer; kwargs...)
end
