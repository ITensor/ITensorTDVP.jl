using DifferentialEquations
using ITensors

# TODO: Define in ITensors.jl
Base.complex(ψ::ITensors.AbstractMPS) = complex.(ψ)
ITensors.contract(ψ::ITensors.AbstractMPS) = prod(ψ)
(A::ITensor)(x) = apply(A, x)

# Helper function to apply the function `f` to the
# ITensor formed by converting the input Vector `v`
# to an ITensor with the indices `inds` and then
# converting back to a vector.
function apply_to_itensor(f, v::Vector, inds)
  return vec(array(permute(f(itensor(v, inds)), inds)))
end

# Represents a time-dependent operator:
#
# H(t) = f[1](t) * H0[1] + f[2](t) * H0[2] + …
#
struct TimeDependentOperator{T}
  f::Vector
  H0::Vector{T}
end
TimeDependentOperator(f::Vector, H0::ProjMPOSum) = TimeDependentOperator(f, H0.pm)
Base.length(H::TimeDependentOperator) = length(H.f)

function Base.:*(c::Number, H::TimeDependentOperator)
  return TimeDependentOperator([t -> c * fₙ(t) for fₙ in H.f], H.H0)
end
Base.:*(H::TimeDependentOperator, c::Number) = c * H

# Calling a `TimeDependentOperator` at a certain time like:
#
# H(t)
#
# Returns a `ScaledOperatorSum` at that time.
(H::TimeDependentOperator)(t::Number) = ScaledOperatorSum([fₙ(t) for fₙ in H.f], H.H0)

# Represents the sum of scaled operators:
#
# H = coefficient[1] * H[1] + coefficient * H[2] + …
#
struct ScaledOperatorSum{S,T}
  coefficients::Vector{S}
  H::T
end
Base.length(H::ScaledOperatorSum) = length(H.coefficients)

# Solve:
# d|ψ(t)⟩/dt = -i H(t) |ψ(t)⟩

# Apply the time dependent Hamiltonian:
#
# H(ψ₀) = coefficient[1] * H[1](ψ₀) + coefficient[2] * H[2](ψ₀) + …
#
# onto ψ₀.
function (H::ScaledOperatorSum)(ψ₀)
  ψ = ITensor(inds(ψ₀))
  for n in 1:length(H)
    ψ += H.coefficients[n] * H.H[n](ψ₀)
  end
  return permute(ψ, inds(ψ₀))
end

function ode_solver(
  H::TimeDependentOperator,
  time_step,
  ψ₀;
  time_step_start=0.0,
  solver_alg=Tsit5(),
  kwargs...,
)
  time_step_stop = time_step_start + time_step
  time_span = (time_step_start, time_step_stop)
  f(u, p, t) = apply_to_itensor(H(t), u, inds(ψ₀))
  u₀ = vec(array(ψ₀))
  prob = ODEProblem(f, u₀, time_span)
  sol = solve(prob, solver_alg; kwargs...)
  uₜ = sol.u[end]
  return itensor(uₜ, inds(ψ₀)), nothing
end

function krylov_solver(
  H::TimeDependentOperator, time_step, ψ₀; time_step_start=0.0, kwargs...
)
  ψₜ, info = exponentiate(H(time_step_start), time_step, ψ₀; kwargs...)
  return ψₜ, info
end

function heisenberg(n; J=1.0, J2=0.0)
  ℋ = OpSum()
  if !iszero(J)
    for j in 1:(n - 1)
      ℋ += J / 2, "S+", j, "S-", j + 1
      ℋ += J / 2, "S-", j, "S+", j + 1
      ℋ += J, "Sz", j, "Sz", j + 1
    end
  end
  if !iszero(J2)
    for j in 1:(n - 2)
      ℋ += J2 / 2, "S+", j, "S-", j + 2
      ℋ += J2 / 2, "S-", j, "S+", j + 2
      ℋ += J2, "Sz", j, "Sz", j + 2
    end
  end
  return ℋ
end
