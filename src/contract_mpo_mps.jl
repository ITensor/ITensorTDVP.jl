function contractmpo_solver(; kwargs...)
  function solver(H, psi0; kws...)
    psi = contract(H,psi0)
    noprime!(psi)
    return psi
  end
  return solver
end

# TODO: Rename this to `contract(::Algorithm"fit",..)` when merging into ITensors.jl

function fit_contract_mpo(A::MPO, psi0::MPS; kwargs...)::MPS
  n = length(A)
  n != length(psi0) &&
    throw(DimensionMismatch("lengths of MPO ($n) and MPS ($(length(psi0))) do not match"))
  if n == 1
    return MPS([A[1] * psi0[1]])
  end

  cutoff::Float64 = get(kwargs, :cutoff, 1e-13)
  maxdim::Int = get(kwargs, :maxdim, maxlinkdim(A) * maxlinkdim(psi0))
  mindim::Int = max(get(kwargs, :mindim, 1), 1)
  normalize::Bool = get(kwargs, :normalize, false)

  any(i -> isempty(i), siteinds(commoninds, A, psi0)) &&
    error("In `contract(A::MPO, x::MPS)`, `A` and `x` must share a set of site indices")

  # In case A and psi0 have the same link indices
  A = sim(linkinds, A)

  t = Inf
  reverse_step = false
  psi = tdvp(contractmpo_solver(; kwargs...), H, t, psi0; reverse_step, kwargs...)

  return psi
end
