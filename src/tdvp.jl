function exponentiate_solver(; kwargs...)
  function solver(H, t, psi0; kws...)
    solver_kwargs = (;
      ishermitian=get(kwargs, :ishermitian, true),
      issymmetric=get(kwargs, :issymmetric, true),
      tol=get(kwargs, :solver_tol, 1E-12),
      krylovdim=get(kwargs, :solver_krylovdim, 30),
      maxiter=get(kwargs, :solver_maxiter, 100),
      verbosity=get(kwargs, :solver_outputlevel, 0),
      eager=true,
    )
    psi, info = exponentiate(H, t, psi0; solver_kwargs...)
    return psi, info
  end
  return solver
end

function applyexp_solver(; kwargs...)
  function solver(H, t, psi0; kws...)
    tol_per_unit_time = get(kwargs, :solver_tol, 1E-8)
    solver_kwargs = (;
      maxiter=get(kwargs, :solver_krylovdim, 30),
      outputlevel=get(kwargs, :solver_outputlevel, 0),
    )
    #applyexp tol is absolute, compute from tol_per_unit_time:
    tol = abs(t) * tol_per_unit_time
    psi, info = applyexp(H, t, psi0; tol, solver_kwargs..., kws...)
    return psi, info
  end
  return solver
end

function tdvp_solver(; kwargs...)
  solver_backend = get(kwargs, :solver_backend, "applyexp")
  if solver_backend == "applyexp"
    return applyexp_solver(; kwargs...)
  elseif solver_backend == "exponentiate"
    return exponentiate_solver(; kwargs...)
  else
    error(
      "solver_backend=$solver_backend not recognized (options are \"applyexp\" or \"exponentiate\")",
    )
  end
end

"""
    tdvp(H::MPO,psi0::MPS,t::Number; kwargs...)
    tdvp(H::MPO,psi0::MPS,t::Number; kwargs...)

    tdvp(Hs::Vector{MPO},psi0::MPS,t::Number; kwargs...)
    tdvp(Hs::Vector{MPO},psi0::MPS,t::Number, sweeps::Sweeps; kwargs...)

Use the time dependent variational principle (TDVP) algorithm
to compute `exp(t*H)*psi0` using an efficient algorithm based
on alternating optimization of the MPS tensors and local Krylov
exponentiation of H.

The version of `tdvp` accepting a
Vector of MPOs, Hs = [H1,H2,H3,...] means that H is defined
as H = H1+H2+H3+...
Note that this sum of MPOs is not actually computed; rather
the set of MPOs [H1,H2,H3,..] is efficiently looped over at 
each step of the algorithm when optimizing the MPS.
                    
Returns:
* `psi::MPS` - time-evolved MPS

Optional keyword arguments:
* `outputlevel::Int = 1` - larger outputlevel values resulting in printing more information and 0 means no output
* `observer` - object implementing the [Observer](@ref observer) interface which can perform measurements and stop early
* `write_when_maxdim_exceeds::Int` - when the allowed maxdim exceeds this value, begin saving tensors to disk to free memory in large calculations
"""
function tdvp(H, t::Number, psi0::MPS; kwargs...)
  return alternating_update(tdvp_solver(; kwargs...), H, t, psi0; kwargs...)
end

# Versions taking alternate argument ordering:

function tdvp(t::Number, H, psi0::MPS; kwargs...)
  return tdvp(H, t, psi0; kwargs...)
end

function tdvp(H, psi0::MPS, t::Number; kwargs...)
  return tdvp(H, t, psi0; kwargs...)
end
