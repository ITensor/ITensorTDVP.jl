
function proj(P::ProjMPS)
  ϕ = prime(linkinds, P.M)
  p = ITensor(1.0)
  !isnothing(lproj(P)) && (p *= lproj(P))
  for j in (P.lpos + 1):(P.rpos - 1)
    p *= dag(ϕ[j])
  end
  !isnothing(rproj(P)) && (p *= rproj(P))
  return dag(p)
end

"""
Compute a solution x to the linear system:

(a₀ + a₁ * A)*x = b

using starting guess x₀. Leaving a₀, a₁
set to their default values solves the 
system A*x = b.
"""
function linsolve(A::MPO, b::MPS, x₀::MPS, a₀::Number=0, a₁::Number=1; reverse_step=false, kwargs...)
  function linsolve_solver(P::ProjMPO_MPS, t, x₀; kws...)
    solver_kwargs = (;
      ishermitian=get(kws, :ishermitian, false),
      tol=get(kws, :solver_tol, 1E-14),
      krylovdim=get(kws, :solver_krylovdim, 30),
      maxiter=get(kws, :solver_maxiter, 100),
      verbosity=get(kws, :solver_verbosity, 0),
    )
    #@show solver_kwargs
    #ITensors.pause()
    A = P.PH
    b = proj(only(P.pm))
    #@show inds(product(A,x₀))
    #@show norm(product(A,x₀)-b)
    #ITensors.pause()
    x, info = KrylovKit.linsolve(A, b, x₀, a₀, a₁; solver_kwargs...)
    return x, nothing
  end
  t = Inf
  P = ProjMPO_MPS(A, [b])
  return tdvp(linsolve_solver, P, t, x₀; reverse_step, kwargs...)
end

