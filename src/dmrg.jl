function eigsolve_solver(;
  solver_which_eigenvalue=:SR,
  ishermitian=true,
  solver_tol=1e-14,
  solver_krylovdim=3,
  solver_maxiter=1,
  solver_verbosity=0,
)
  function solver(H, t, psi0)
    howmany = 1
    which = solver_which_eigenvalue
    vals, vecs, info = eigsolve(
      H,
      psi0,
      howmany,
      which;
      ishermitian=true,
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

function dmrg(H, psi0::MPS; kwargs...)
  t = Inf # DMRG is TDVP with an infinite timestep and no reverse step
  reverse_step = false
  psi = tdvp(eigsolve_solver(; kwargs...), H, t, psi0; reverse_step, kwargs...)
  return psi
end
