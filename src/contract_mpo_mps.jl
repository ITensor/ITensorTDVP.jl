using ITensors:
  ITensors,
  Index,
  ITensor,
  @Algorithm_str,
  commoninds,
  contract,
  hasind,
  linkinds,
  replace_siteinds!,
  sim,
  siteinds

function contractmpo_solver(; kwargs...)
  function solver(reduced_operator, psi; kws...)
    reduced_state = ITensor(true)
    for j in (reduced_operator.lpos + 1):(reduced_operator.rpos - 1)
      reduced_state *= reduced_operator.input_state[j]
    end
    reduced_state = contract(reduced_operator, reduced_state)
    return reduced_state, nothing
  end
  return solver
end

# `init_mps` is for backwards compatibility.
function ITensors.contract(
  ::Algorithm"fit",
  operator::MPO,
  input_state::MPS;
  init=input_state,
  init_mps=init,
  kwargs...,
)::MPS
  n = length(operator)
  n != length(input_state) && throw(
    DimensionMismatch("lengths of MPO ($n) and MPS ($(length(input_state))) do not match")
  )
  if n == 1
    return MPS([operator[1] * input_state[1]])
  end
  any(i -> isempty(i), siteinds(commoninds, operator, input_state)) && error(
    "In `contract(operator::MPO, x::MPS)`, `operator` and `x` must share a set of site indices",
  )
  # In case operator and input_state have the same link indices
  operator = sim(linkinds, operator)
  # Fix site and link inds of init
  init = deepcopy(init)
  init = sim(linkinds, init)
  siteinds_operator = siteinds(operator)
  ti = Vector{Index}(undef, n)
  for j in 1:n
    for i in siteinds_operator[j]
      if !hasind(input_state[j], i)
        ti[j] = i
        break
      end
    end
  end
  replace_siteinds!(init, ti)
  reduced_operator = ProjMPOApply(input_state, operator)
  psi = alternating_update(
    contractmpo_solver(; kwargs...), reduced_operator, init; kwargs...
  )
  return psi
end
