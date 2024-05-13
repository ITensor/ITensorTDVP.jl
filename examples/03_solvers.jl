using ITensors: ITensor, array, inds, itensor
using ITensorMPS: TimeDependentSum
using KrylovKit: exponentiate
using OrdinaryDiffEq: ODEProblem, Tsit5, solve

function to_vec(x::ITensor)
  function to_itensor(x_vec)
    return itensor(x_vec, inds(x))
  end
  return vec(array(x)), to_itensor
end

function ode_updater(
  H::TimeDependentSum,
  ψ₀;
  time_step,
  current_time=zero(time_step),
  outputlevel=0,
  updater_alg=Tsit5(),
  kwargs...,
)
  if outputlevel ≥ 3
    println("    In ODE updater, current_time = $current_time, time_step = $time_step")
  end

  time_span = (current_time, current_time + time_step)
  u₀, to_itensor = to_vec(ψ₀)
  f(ψ::ITensor, p, t) = H(t)(ψ)
  f(u::Vector, p, t) = to_vec(f(to_itensor(u), p, t))[1]
  prob = ODEProblem(f, u₀, time_span)
  sol = solve(prob, updater_alg; kwargs...)
  uₜ = sol.u[end]
  return to_itensor(uₜ), nothing
end

function ode_updater(f⃗, H⃗₀, ψ₀; time_step, kwargs...)
  return ode_updater(-im * TimeDependentSum(f⃗, H⃗₀), ψ₀; time_step, kwargs...)
end

function krylov_updater(
  H::TimeDependentSum, ψ₀; time_step, current_time=zero(time_step), outputlevel=0, kwargs...
)
  if outputlevel ≥ 3
    println("    In Krylov updater, current_time = $current_time, time_step = $time_step")
  end
  ψₜ, info = exponentiate(H(current_time), time_step, ψ₀; kwargs...)
  return ψₜ, info
end

function krylov_updater(f⃗, H⃗₀, ψ₀; time_step, kwargs...)
  return krylov_updater(-im * TimeDependentSum(f⃗, H⃗₀), ψ₀; time_step, kwargs...)
end
