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

function contract_operator_state_updater(operator, init; internal_kwargs)
  # TODO: Use `contract(operator)`.
  state = ITensor(true)
  for j in (operator.lpos + 1):(operator.rpos - 1)
    state *= operator.input_state[j]
  end
  state = contract(operator, state)
  return state, (;)
end

# `init_mps` is for backwards compatibility.
function ITensors.contract(
  ::Algorithm"fit",
  operator::MPO,
  input_state::MPS;
  init=input_state,
  init_mps=init,
  kwargs...,
)
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
  reduced_operator = ReducedContractProblem(input_state, operator)
  return alternating_update(
    reduced_operator, init; updater=contract_operator_state_updater, kwargs...
  )
end
