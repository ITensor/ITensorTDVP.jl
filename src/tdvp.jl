using ITensors: Algorithm, MPO, MPS, @Algorithm_str
using KrylovKit: exponentiate

# Select solver function
solver_function(solver_backend::String) = solver_function(Algorithm(solver_backend))
solver_function(::Algorithm"exponentiate") = exponentiate
solver_function(::Algorithm"applyexp") = applyexp
function solver_function(solver_backend::Algorithm)
  return error(
    "solver_backend=$(String(solver_backend)) not recognized (only \"exponentiate\" is supported)",
  )
end

function tdvp_solver(
  f::Function;
  ishermitian,
  issymmetric,
  solver_tol,
  solver_krylovdim,
  solver_maxiter,
  solver_outputlevel,
)
  function solver(operator, t, init; current_time, outputlevel)
    return f(
      operator,
      t,
      init;
      ishermitian,
      issymmetric,
      tol=solver_tol,
      krylovdim=solver_krylovdim,
      maxiter=solver_maxiter,
      verbosity=solver_outputlevel,
      eager=true,
    )
  end
  return solver
end

function time_step_and_nsteps(t, time_step::Nothing, nsteps::Nothing)
  return error("Must specify either `time_step`, `nsteps`, or both.")
end

function time_step_and_nsteps(t, time_step::Nothing, nsteps)
  return t / nsteps, nsteps
end

function time_step_and_nsteps(t, time_step, nsteps::Nothing)
  nsteps, rem = divrem(t, time_step)
  if rem ≉ 0
    return error("`t / time_step = $t / $time_step = $(t / time_step)` must be an integer.")
  end
  return time_step, Int(nsteps)
end

function time_step_and_nsteps(t, time_step, nsteps)
  if time_step * nsteps ≠ t
    return error(
      "`t = $t`, `time_step = $time_step`, and `nsteps = $nsteps` must satisfy `time_steps * nsteps == t`, while `time_steps * nsteps = $time_steps * $nsteps`.",
    )
  end
  return time_step, nsteps
end

"""
    tdvp(operator, t::Number, init::MPS; time_step, nsteps, kwargs...)

Use the time dependent variational principle (TDVP) algorithm
to compute `exp(t * operator) * init` using an efficient algorithm based
on alternating optimization of the MPS tensors and local Krylov
exponentiation of `operator`.

Specify one of `time_step` or `nsteps`. If they are both specified, they
must satisfy `time_step * nsteps == t`.

Returns:
* `state::MPS` - time-evolved MPS

"""
function tdvp(
  operator,
  t::Number,
  init::MPS;
  reverse_step=true,
  time_step=nothing,
  time_start=0,
  nsweeps=nothing,
  nsteps=nsweeps,
  ishermitian=default_ishermitian(),
  issymmetric=default_issymmetric(),
  solver_backend=default_tdvp_solver_backend(),
  solver_function=solver_function(solver_backend),
  solver_tol=default_solver_tol(solver_function),
  solver_krylovdim=default_solver_krylovdim(solver_function),
  solver_maxiter=default_solver_maxiter(solver_function),
  solver_outputlevel=default_solver_outputlevel(solver_function),
  kwargs...,
)
  @show t, time_step, nsteps
  time_step, nsteps = time_step_and_nsteps(t, time_step, nsteps)
  @show time_step, nsweeps
  return alternating_update(
    tdvp_solver(
      solver_function;
      ishermitian,
      issymmetric,
      solver_tol,
      solver_krylovdim,
      solver_maxiter,
      solver_outputlevel,
    ),
    operator,
    init;
    reverse_step,
    nsweeps=nsteps,
    time_step,
    kwargs...,
  )
end

## function tdvp(t::Number, H, psi0::MPS; kwargs...)
##   return tdvp(H, t, psi0; kwargs...)
## end
## 
## function tdvp(H, psi0::MPS, t::Number; kwargs...)
##   return tdvp(H, t, psi0; kwargs...)
## end

## """
##     tdvp(H::MPO,psi0::MPS,t::Number; kwargs...)
##     tdvp(H::MPO,psi0::MPS,t::Number; kwargs...)
## 
## Use the time dependent variational principle (TDVP) algorithm
## to compute `exp(t*H)*psi0` using an efficient algorithm based
## on alternating optimization of the MPS tensors and local Krylov
## exponentiation of H.
## 
## Returns:
## * `psi::MPS` - time-evolved MPS
## 
## Optional keyword arguments:
## * `outputlevel::Int = 1` - larger outputlevel values resulting in printing more information and 0 means no output
## * `observer` - object implementing the [Observer](@ref observer) interface which can perform measurements and stop early
## * `write_when_maxdim_exceeds::Int` - when the allowed maxdim exceeds this value, begin saving tensors to disk to free memory in large calculations
## """
## function tdvp(solver, H::MPO, t::Number, psi0::MPS; kwargs...)
##   return alternating_update(solver, H, t, psi0; kwargs...)
## end

## function tdvp(solver, t::Number, H, psi0::MPS; kwargs...)
##   return tdvp(solver, H, t, psi0; kwargs...)
## end

## function tdvp(solver, H, psi0::MPS, t::Number; kwargs...)
##   return tdvp(solver, H, t, psi0; kwargs...)
## end

## """
##     tdvp(Hs::Vector{MPO},psi0::MPS,t::Number; kwargs...)
##     tdvp(Hs::Vector{MPO},psi0::MPS,t::Number, sweeps::Sweeps; kwargs...)
## 
## Use the time dependent variational principle (TDVP) algorithm
## to compute `exp(t*H)*psi0` using an efficient algorithm based
## on alternating optimization of the MPS tensors and local Krylov
## exponentiation of H.
## 
## This version of `tdvp` accepts a representation of H as a
## Vector of MPOs, Hs = [H1,H2,H3,...] such that H is defined
## as H = H1+H2+H3+...
## Note that this sum of MPOs is not actually computed; rather
## the set of MPOs [H1,H2,H3,..] is efficiently looped over at
## each step of the algorithm when optimizing the MPS.
## 
## Returns:
## * `psi::MPS` - time-evolved MPS
## """
## function tdvp(solver, Hs::Vector{MPO}, t::Number, psi0::MPS; kwargs...)
##   return alternating_update(solver, Hs, t, psi0; kwargs...)
## end
