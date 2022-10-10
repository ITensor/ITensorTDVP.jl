
"""
Compute a solution x to the linear system:

(a₀ + a₁ * A)*x = b

using starting guess x₀. Leaving a₀, a₁
set to their default values solves the 
system A*x = b.
"""
function linsolve(A::MPO, b::MPS, x₀::MPS, a₀::Number=0, a₁::Number=1; reverse_step=false, kwargs...)
  function linsolve_solver(P::ProjMPO_MPS2, t, x₀; kws...)
    solver_kwargs = (;
      ishermitian=kws[:ishermitian],
      tol=kws[:solver_tol],
      krylovdim=kws[:solver_krylovdim],
      maxiter=kws[:solver_maxiter],
      verbosity=kws[:solver_verbosity]
    )
    b = only(proj_mps(P))
    x, info = KrylovKit.linsolve(P, b, x₀, a₀, a₁; solver_kwargs...)
    return x, nothing
  end

  # Set appropriate linsolve defaults
  solver_default_kwargs = (;
    solver_tol=get(kwargs, :solver_tol, 1E-14),
    solver_krylovdim=get(kwargs, :solver_krylovdim, 30),
    solver_maxiter=get(kwargs, :solver_maxiter, 100)
   )

  t = Inf
  P = ProjMPO_MPS2(A, b)
  return tdvp(linsolve_solver, P, t, x₀; reverse_step, solver_default_kwargs..., kwargs...)
end

