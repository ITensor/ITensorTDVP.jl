using ITensors: @Algorithm_str, Algorithm

default_solver_tol(::typeof(exponentiate)) = 1e-12

function tdvp_solver(
  f::typeof(exponentiate);
  ishermitian=true,
  issymmetric=true,
  solver_tol=1e-12,
  solver_krylovdim=30,
  solver_maxiter=100,
  solver_outputlevel=0,
)
  function solver(H, t, psi0; current_time, outputlevel)
    psi, info = f(
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

function tdvp_solver(
  f::typeof(applyexp); solver_tol=1e-8, solver_krylovdim=30, solver_outputlevel=0
)
  function solver(H, t, psi0)
    # applyexp tol is absolute, compute from solver_tol:
    tol = abs(t) * solver_tol
    psi, info = f(H, t, psi0; tol, maxiter=solver_krylovdim, outputlevel=solver_outputlevel)
    return psi, info
  end
  return solver
end

## function tdvp_solver(; solver_backend="exponentiate", kwargs...)
##   if solver_backend == "applyexp"
##     return applyexp_solver(; kwargs...)
##   elseif solver_backend == "exponentiate"
##     return exponentiate_solver(; kwargs...)
##   else
##     error(
##       "solver_backend=$solver_backend not recognized (options are \"applyexp\" or \"exponentiate\")",
##     )
##   end
## end

solver_function(solver_backend::String) = solver_function(Algorithm(solver_backend))
solver_function(solver_backend::Algorithm"exponentiate") = exponentiate
solver_function(solver_backend::Algorithm"applyexp") = applyexp
function solver_function(solver_backend::Algorithm)
  return error(
    "solver_backend=$(String(solver_backend)) not recognized (options are \"applyexp\" or \"exponentiate\")",
  )
end

function tdvp(
  H,
  t::Number,
  psi0::MPS;
  solver_backend=default_tdvp_solver_backend(),
  solver_tol=default_solver_tol(solver_function(solver_backend)),
  solver_krylovdim=default_solver_krylovdim(solver_function(solver_backend)),
  solver_outputlevel=default_solver_outputlevel(solver_function(solver_backend)),
  kwargs...,
)
  return tdvp(
    tdvp_solver(; solver_backend, solver_tol, solver_krylovdim, solver_outputlevel),
    H,
    t,
    psi0;
    kwargs...,
  )
end

function tdvp(t::Number, H, psi0::MPS; kwargs...)
  return tdvp(H, t, psi0; kwargs...)
end

function tdvp(H, psi0::MPS, t::Number; kwargs...)
  return tdvp(H, t, psi0; kwargs...)
end
