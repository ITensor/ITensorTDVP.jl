function exponentiate_solver(; kwargs...)
  solver_kwargs = (;
    ishermitian=get(kwargs, :ishermitian, true),
    issymmetric=get(kwargs, :issymmetric, true),
    tol=get(kwargs, :solver_tol, 1E-12),
    krylovdim=get(kwargs, :solver_krylovdim, 30),
    maxiter=get(kwargs, :solver_maxiter, 100),
    verbosity=get(kwargs, :solver_outputlevel, 0),
    eager=true,
  )
  function solver(H, t, psi0; kws...)
    psi, info = exponentiate(H, t, psi0; solver_kwargs..., kws...)
    return psi, info
  end
  return solver
end

function applyexp_solver(; kwargs...)
  tol_per_unit_time = get(kwargs, :solver_tol, 1E-8)
  solver_kwargs = (;
    maxiter=get(kwargs, :solver_krylovdim, 30),
    outputlevel=get(kwargs, :solver_outputlevel, 0),
  )
  function solver(H, t, psi0; kws...)
    #apply_exp tol is absolute, compute from tol_per_unit_time:
    tol = abs(t) * tol_per_unit_time
    psi, info = apply_exp(H, t, psi0; tol, solver_kwargs..., kws...)
    return psi, info
  end
  return solver
end

function tdvp_solver(; kwargs...)
  solver_backend = get(kwargs, :solver_backend, "exponentiate")
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

function tdvp(H, t::Number, psi0::MPS; kwargs...)
  return tdvp(tdvp_solver(; kwargs...), H, t, psi0; kwargs...)
end
