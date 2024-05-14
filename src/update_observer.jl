function update_observer!(observer; kwargs...)
  return error("Not implemented")
end

struct EmptyObserver end
update_observer!(observer::EmptyObserver; kwargs...) = observer

using ITensors.ITensorMPS: ITensorMPS
function update_observer!(observer::ITensorMPS.AbstractObserver; kwargs...)
  return ITensorMPS.measure!(observer; kwargs...)
end

struct ValuesObserver{Values<:NamedTuple}
  values::Values
end
function update_observer!(observer::ValuesObserver; kwargs...)
  for key in keys(observer.values)
    observer.values[key][] = kwargs[key]
  end
  return observer
end
values_observer(; kwargs...) = ValuesObserver(NamedTuple(kwargs))

struct ComposedObservers{Observers<:Tuple}
  observers::Observers
end
compose_observers(observers...) = ComposedObservers(observers)
function update_observer!(observer::ComposedObservers; kwargs...)
  for observerᵢ in observer.observers
    update_observer!(observerᵢ; kwargs...)
  end
  return observer
end
