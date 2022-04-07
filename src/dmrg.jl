function eigsolve_solver(; kwargs...)
  howmany = 1
  which = get(kwargs, :solver_which_eigenvalue, :SR)
  solver_kwargs = (;
    ishermitian=get(kwargs, :ishermitian, true),
    tol=get(kwargs, :solver_tol, 1E-14),
    krylovdim=get(kwargs, :solver_krylovdim, 3),
    maxiter=get(kwargs, :solver_maxiter, 1),
    verbosity=get(kwargs, :solver_verbosity, 0),
  )
  function solver(H, t, psi0; kws...)
    vals, vecs, info = eigsolve(H, psi0, howmany, which; solver_kwargs..., kws...)
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

function dmrg(H::MPO, psi0::MPS; kwargs...)
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  H = ITensors.permute(H, (linkind, siteinds, linkind))
  PH = ProjMPO(H)
  return dmrg(PH, psi0; kwargs...)
end
