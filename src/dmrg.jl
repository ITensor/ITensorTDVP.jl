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
    return vecs[1], (; info, eigval=vals[1])
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
  (observer!)=default_observer!(),
  kwargs...,
)
  reverse_step = false
  info_ref! = Ref{Any}()
  info_observer! = values_observer(; info=info_ref!)
  observer! = compose_observers(observer!, info_observer!)
  psi = alternating_update(
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
    psi0;
    reverse_step,
    observer!,
    kwargs...,
  )
  return info_ref![].eigval, psi
end
