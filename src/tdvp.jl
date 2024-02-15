using ITensors: @Algorithm_str, Algorithm

# Select solver function
solver_function(solver_backend::String) = solver_function(Algorithm(solver_backend))
solver_function(::Algorithm"exponentiate") = exponentiate
function solver_function(solver_backend::Algorithm)
  return error(
    "solver_backend=$(String(solver_backend)) not recognized (only \"exponentiate\" is supported)",
  )
end

# Kept for backwards compatibility
function solver_function(::Algorithm"applyexp")
  println(
    "Warning: the `solver_backend` option `\"applyexp\"` in `tdvp` has been removed. `\"exponentiate\"` will be used instead. To remove this warning, don't specify the `solver_backend` keyword argument.",
  )
  return solver_function(Algorithm"exponentiate"())
end

function tdvp_solver(
  f::typeof(exponentiate);
  ishermitian,
  issymmetric,
  solver_tol,
  solver_krylovdim,
  solver_maxiter,
  solver_outputlevel,
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

function tdvp(
  H,
  t::Number,
  psi0::MPS;
  ishermitian=default_ishermitian(),
  issymmetric=default_issymmetric(),
  solver_backend=default_tdvp_solver_backend(),
  solver_tol=default_solver_tol(solver_function(solver_backend)),
  solver_krylovdim=default_solver_krylovdim(solver_function(solver_backend)),
  solver_maxiter=default_solver_maxiter(solver_function(solver_backend)),
  solver_outputlevel=default_solver_outputlevel(solver_function(solver_backend)),
  kwargs...,
)
  return tdvp(
    tdvp_solver(
      solver_function(solver_backend);
      ishermitian,
      issymmetric,
      solver_tol,
      solver_krylovdim,
      solver_maxiter,
      solver_outputlevel,
    ),
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
