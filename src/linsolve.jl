using ITensors: MPO, MPS
using KrylovKit: KrylovKit, linsolve

"""
Compute a solution x to the linear system:

(a₀ + a₁ * A)*x = b

using starting guess x₀. Leaving a₀, a₁
set to their default values solves the 
system A*x = b.

To adjust the balance between accuracy of solution
and speed of the algorithm, it is recommed to first try
adjusting the solver keyword arguments as descibed below.

Keyword arguments:
  - `nsweeps`, `cutoff`, `maxdim`, etc. (like for other MPO/MPS solvers).
  - `solver_kwargs=(;)` - a `NamedTuple` containing keyword arguments that will get forwarded to the local solver,
    in this case `KrylovKit.linsolve` which is a GMRES linear solver. For example:
    ```juli
    linsolve(A, b, x; maxdim=100, cutoff=1e-8, nsweeps=10, solver_kwargs=(; ishermitian=true, tol=1e-6, maxiter=20, krylovdim=30))
    ```
    See `KrylovKit.jl` documentation for more details on available keyword arguments.
"""
function KrylovKit.linsolve(
  operator::MPO,
  constant_term::MPS,
  init::MPS,
  a₀::Number=false,
  a₁::Number=true;
  solver_kwargs=(;),
  kwargs...,
)
  function linsolve_solver(
    reduced_problem::ReducedLinearProblem, init; current_time, time_step, outputlevel
  )
    constant_term = dag(only(reduced_constant_terms(reduced_problem)))
    x, info = linsolve(reduced_problem, constant_term, init, a₀, a₁; solver_kwargs...)
    return x, nothing
  end
  reduced_problem = ReducedLinearProblem(operator, constant_term)
  return alternating_update(linsolve_solver, reduced_problem, init; kwargs...)
end
