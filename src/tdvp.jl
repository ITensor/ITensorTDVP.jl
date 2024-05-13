using ITensors: Algorithm, MPO, MPS, @Algorithm_str
using KrylovKit: exponentiate

## # Select updater function
## updater_function(updater_backend::String) = updater_function(Algorithm(updater_backend))
## updater_function(::Algorithm"exponentiate") = exponentiate
## updater_function(::Algorithm"applyexp") = applyexp
## function updater_function(updater_backend::Algorithm)
##   return error(
##     "updater_backend=$(String(updater_backend)) not recognized (only \"exponentiate\" is supported)",
##   )
## end

## function tdvp_updater(
##   f::Function;
##   ishermitian,
##   issymmetric,
##   updater_tol,
##   updater_krylovdim,
##   updater_maxiter,
##   updater_outputlevel,
## )
##   function updater(operator, init; current_time, time_step, outputlevel)
##     return f(
##       operator,
##       time_step,
##       init;
##       ishermitian,
##       issymmetric,
##       tol=updater_tol,
##       krylovdim=updater_krylovdim,
##       maxiter=updater_maxiter,
##       verbosity=updater_outputlevel,
##       eager=true,
##     )
##   end
##   return updater
## end

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

function default_tdvp_updater()
  return tdvp_updater(
    updater_function;
    ishermitian,
    issymmetric,
    updater_tol,
    updater_krylovdim,
    updater_maxiter,
    updater_outputlevel,
  )
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
  time_start=zero(t),
  nsweeps=nothing,
  nsteps=nsweeps,
  ishermitian=default_ishermitian(),
  issymmetric=default_issymmetric(),
  updater=default_tdvp_updater(),
  updater_backend=default_tdvp_updater_backend(),
  updater_function=updater_function(updater_backend),
  updater_tol=default_updater_tol(updater_function),
  updater_krylovdim=default_updater_krylovdim(updater_function),
  updater_maxiter=default_updater_maxiter(updater_function),
  updater_outputlevel=default_updater_outputlevel(updater_function),
  kwargs...,
)
  time_step, nsteps = time_step_and_nsteps(t, time_step, nsteps)
  return alternating_update(
    updater, operator, init; reverse_step, nsweeps=nsteps, time_start, time_step, kwargs...
  )
end
