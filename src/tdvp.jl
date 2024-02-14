function exponentiate_solver(;
  ishermitian=true,
  issymmetric=true,
  solver_tol=1e-12,
  solver_krylovdim=30,
  solver_maxiter=100,
  solver_outputlevel=0,
  kwargs...,
)
  function solver(H, t, psi0)
    psi, info = exponentiate(
      H,
      t,
      psi0;
      ishermitian,
      issymmetric,
      tol=solver_tol,
      krylovdim=solver_krylovdim,
      maxiter=solver_maxiter,
      verbosity=solver_outputlevel,
      eager=true,
    )
    return psi, info
  end
  return solver
end

function applyexp_solver(;
  solver_tol=1e-8,
  solver_krylovdim=30,
  solver_outputlevel=0,
)
  function solver(H, t, psi0)
    # applyexp tol is absolute, compute from solver_tol:
    tol = abs(t) * solver_tol
    psi, info = applyexp(
      H,
      t,
      psi0;
      tol,
      maxiter=solver_krylovdim,
      outputlevel=solver_outputlevel,
    )
    return psi, info
  end
  return solver
end

function tdvp_solver(;
  solver_backend="exponentiate",
  kwargs...,
)
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

function tdvp(t::Number, H, psi0::MPS; kwargs...)
  return tdvp(H, t, psi0; kwargs...)
end

function tdvp(H, psi0::MPS, t::Number; kwargs...)
  return tdvp(H, t, psi0; kwargs...)
end
