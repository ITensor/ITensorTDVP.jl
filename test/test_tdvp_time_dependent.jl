using DifferentialEquations
using ITensors
using ITensorTDVP
using KrylovKit
using LinearAlgebra
using Test

include(joinpath(pkgdir(ITensorTDVP), "examples", "03_models.jl"))
include(joinpath(pkgdir(ITensorTDVP), "examples", "03_solvers.jl"))

# Functions need to be defined in global scope (outside
# of the @testset macro)

function ode_solver(H⃗₀, time_step, ψ₀; kwargs...)
  ω₁ = typeof(time_step)(0.1)
  ω₂ = typeof(time_step)(0.2)
  ode_alg = Tsit5()
  tol = √eps(real(time_step))
  ode_kwargs = (; reltol=tol, abstol=tol)
  ω⃗ = [ω₁, ω₂]
  f⃗ = [t -> cos(ω * t) for ω in ω⃗]
  return ode_solver(
    -im * TimeDependentSum(f⃗, H⃗₀),
    time_step,
    ψ₀;
    solver_alg=ode_alg,
    ode_kwargs...,
    kwargs...,
  )
end

function krylov_solver(H⃗₀, time_step, ψ₀; kwargs...)
  tol = √eps(real(time_step))
  krylov_kwargs = (; tol, eager=true)
  return krylov_solver(
    -im * TimeDependentSum(f⃗, H⃗₀), time_step, ψ₀; krylov_kwargs..., kwargs...
  )
end

@testset "Time dependent Hamiltonian (eltype=$elt)" for elt in (
  Float32, Float64, Complex{Float32}, Complex{Float64}
)
  n = 4
  J₁ = elt(1)
  J₂ = elt(0.1)
  time_step = elt(0.1)
  time_stop = elt(1)
  nsite = 2
  maxdim = 100
  cutoff = √(eps(elt))
  s = siteinds("S=1/2", n)
  ℋ₁₀ = heisenberg(n; J=J₁, J2=elt(0))
  ℋ₂₀ = heisenberg(n; J=elt(0), J2=J₂)
  ℋ⃗₀ = [ℋ₁₀, ℋ₂₀]
  H⃗₀ = [MPO(elt, ℋ₀, s) for ℋ₀ in ℋ⃗₀]
  ψ₀ = complex.(MPS(elt, s, j -> isodd(j) ? "↑" : "↓"))
  ψₜ_ode = tdvp(ode_solver, H⃗₀, time_stop, ψ₀; time_step, maxdim, cutoff, nsite)
  ψₜ_krylov = tdvp(krylov_solver, H⃗₀, time_stop, ψ₀; time_step, cutoff, nsite)
  ψₜ_full, _ = ode_solver(prod.(H⃗₀), time_stop, prod(ψ₀))

  @test ITensors.scalartype(ψ₀) == elt
  @test ITensors.scalartype(ψₜ_ode) == elt
  @test ITensors.scalartype(ψₜ_krylov) == elt
  @test ITensors.scalartype(ψₜ_full) == elt
  @test norm(ψ₀) ≈ 1
  @test norm(ψₜ_ode) ≈ 1
  @test norm(ψₜ_krylov) ≈ 1
  @test norm(ψₜ_full) ≈ 1

  ode_err = norm(prod(ψₜ_ode) - ψₜ_full)
  krylov_err = norm(prod(ψₜ_krylov) - ψₜ_full)

  @test krylov_err > ode_err
  @test ode_err < 1e-3
  @test krylov_err < 1e-3
end

nothing
