module ITensorTDVPObserversExt
using Observers: Observers
using Observers.DataFrames: AbstractDataFrame
using ITensorTDVP: ITensorTDVP

function ITensorTDVP.update_observer!(observer::AbstractDataFrame; kwargs...)
  return Observers.update!(observer; kwargs...)
end
end
