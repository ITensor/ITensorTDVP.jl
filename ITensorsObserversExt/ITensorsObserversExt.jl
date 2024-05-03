module ITensorsObserversExt
using Observers: Observers
using Observers.DataFrames: AbstractDataFrame
using ITensors.ITensorTDVP: ITensorTDVP

function ITensorTDVP.update_observer!(observer::AbstractDataFrame; kwargs...)
  return Observers.update!(observer; kwargs...)
end
end
