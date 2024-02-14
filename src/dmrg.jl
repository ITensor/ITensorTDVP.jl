function dmrg_solver(
  f::typeof(eigsolve);
  solver_which_eigenvalue,
  ishermitian,
  solver_tol,
  solver_krylovdim,
  solver_maxiter,
  solver_verbosity,
)
  function solver(H, t, psi0; current_time, outputlevel)
    howmany = 1
    which = solver_which_eigenvalue
    vals, vecs, info = f(
      H,
      psi0,
      howmany,
      which;
      ishermitian=default_ishermitian(),
      tol=solver_tol,
      krylovdim=solver_krylovdim,
      maxiter=solver_maxiter,
      verbosity=solver_verbosity,
    )
    psi = vecs[1]
    return psi, info
  end
  return solver
end

function dmrg(
  H,
  psi0::MPS;
  ishermitian=default_ishermitian(),
  solver_which_eigenvalue=default_solver_which_eigenvalue(eigsolve),
  solver_tol=default_solver_tol(eigsolve),
  solver_krylovdim=default_solver_krylovdim(eigsolve),
  solver_maxiter=default_solver_maxiter(eigsolve),
  solver_verbosity=default_solver_verbosity(),
  kwargs...,
)
  t = Inf # DMRG is TDVP with an infinite timestep and no reverse step
  reverse_step = false
  psi = tdvp(
    dmrg_solver(
      eigsolve;
      solver_which_eigenvalue,
      ishermitian,
      solver_tol,
      solver_krylovdim,
      solver_maxiter,
      solver_verbosity,
    ),
    H,
    t,
    psi0;
    reverse_step,
    kwargs...,
  )
  return psi
end
