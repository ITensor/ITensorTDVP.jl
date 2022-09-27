function contractmpo_solver(; kwargs...)
  function solver(PH, t, psi; kws...)
    j = PH.lpos + 1
    v = ITensor(1.0)
    for j in (PH.lpos + 1):(PH.rpos - 1)
      v *= PH.psi0[j]
    end
    Hpsi0 = contract(PH, v)
    noprime!(Hpsi0)
    return Hpsi0, nothing
  end
  return solver
end

# TODO: Rename this to `contract(::Algorithm"fit",..)` when merging into ITensors.jl

function fit_contract_mpo(A::MPO, psi0::MPS; nsweeps=1, kwargs...)::MPS
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
  PH = ProjMPOApply(psi0, A)
  psi = tdvp(contractmpo_solver(; kwargs...), PH, t, psi0; nsweeps, reverse_step, kwargs...)

  return psi
end
