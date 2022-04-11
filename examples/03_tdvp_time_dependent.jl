using DifferentialEquations
using ITensors
using ITensorTDVP
using KrylovKit
using LinearAlgebra

include("03_utils.jl")

# Time dependent Hamiltonian is:
# H(t) = H₁(t) + H₂(t) + …
#      = f₁(t) H₁(0) + f₂(t) H₂(0) + …
#      = cos(ω₁t) H₁(0) + cos(ω₂t) H₂(0) + …

ω₁ = 0.2
ω₂ = 0.4

ω⃗ = [ω₁, ω₂]
f⃗ = [t -> cos(ω * t) for ω in ω⃗]

time_stop = 1.0
time_step = 0.2

# Number of sites
n = 4

# Nearest and next-nearest neighbor
# Heisenberg couplings.
J₁ = 1.0
J₂ = 0.1

# H₀ = H(0) = H₁(0) + H₂(0) + …
ℋ₁₀ = heisenberg(n; J=J₁, J2=0.0)
ℋ₂₀ = heisenberg(n; J=0.0, J2=J₂)
ℋ⃗₀ = [ℋ₁₀, ℋ₂₀]

s = siteinds("S=1/2", n)

H⃗₀ = [MPO(ℋ₀, s) for ℋ₀ in ℋ⃗₀]

# Initial state, ψ₀ = ψ(0)
# Initialize as complex since that is what DifferentialEquations.jl
# expects.
ψ₀ = complex(MPS(s, j -> isodd(j) ? "↑" : "↓"))

@show norm(ψ₀)

cutoff = 1e-8
# nsite-update TDVP
nsite = 2

#
# Use an ODE solver
#

ode_alg = Tsit5()
ode_kwargs = (; reltol=1e-8, abstol=1e-8)
function ode_solver(H⃗₀, time_step, ψ₀; kwargs...)
  return ode_solver(
    -im * TimeDependentOperator(f⃗, H⃗₀),
    time_step,
    ψ₀;
    solver_alg=ode_alg,
    ode_kwargs...,
    kwargs...,
  )
end

ψₜ_ode = tdvp(ode_solver, H⃗₀, time_stop, ψ₀; time_step, cutoff, nsite)

@show norm(ψₜ_ode)

#
# Use a Krylov exponentiation solver
#

krylov_kwargs = (; tol=1e-8, eager=true)
function krylov_solver(H⃗₀, time_step, ψ₀; kwargs...)
  return krylov_solver(-im * TimeDependentOperator(f⃗, H⃗₀), time_step, ψ₀; krylov_kwargs..., kwargs...)
end

ψₜ_krylov = tdvp(krylov_solver, H⃗₀, time_stop, ψ₀; time_step, cutoff, nsite)

@show norm(ψₜ_krylov)
@show norm(ψₜ_ode - ψₜ_krylov)

# Solve full problem with ODE solver
ψₜ_full, _ = ode_solver(contract.(H⃗₀), time_stop, contract(ψ₀))

@show norm(ψₜ_full)
@show norm(contract(ψₜ_ode) - ψₜ_full)
@show norm(contract(ψₜ_krylov) - ψₜ_full)
