@eval module $(gensym())
# These tests take too long to compile, skip for now.
using ITensors: ITensors, MPO, MPS, siteinds
using ITensorTDVP: ITensorTDVP, tdvp
using LinearAlgebra: norm
using Test: @test, @testset

include(joinpath(pkgdir(ITensorTDVP), "examples", "03_models.jl"))
include(joinpath(pkgdir(ITensorTDVP), "examples", "03_solvers.jl"))

@testset "Time dependent Hamiltonian (eltype=$elt, conserve_qns=$conserve_qns)" for elt in
                                                                                    (
    Float32, Float64, Complex{Float32}, Complex{Float64}
  ),
  conserve_qns in [false, true]

  # Frequencies
  ω₁ = 0.1
  ω₂ = 0.2

  function ode_solver_f⃗(H⃗₀, time_step, ψ₀; kwargs...)
    ω⃗ = typeof(time_step)[ω₁, ω₂]
    f⃗ = [t -> cos(ω * t) for ω in ω⃗]
    tol = √eps(real(time_step))
    return ode_solver(
      f⃗, H⃗₀, time_step, ψ₀; solver_alg=Tsit5(), reltol=tol, abstol=tol, kwargs...
    )
  end

  function krylov_solver_f⃗(H⃗₀, time_step, ψ₀; kwargs...)
    ω⃗ = typeof(time_step)[ω₁, ω₂]
    f⃗ = [t -> cos(ω * t) for ω in ω⃗]
    tol = √eps(real(time_step))
    return krylov_solver(f⃗, H⃗₀, time_step, ψ₀; tol, eager=true, kwargs...)
  end

  n = 4
  J₁ = elt(1)
  J₂ = elt(0.1)
  time_step = real(elt)(0.1)
  time_stop = real(elt)(1)
  nsite = 2
  maxdim = 100
  cutoff = √(eps(real(elt)))
  s = siteinds("S=1/2", n)
  ℋ₁₀ = heisenberg(n; J=J₁, J2=zero(elt))
  ℋ₂₀ = heisenberg(n; J=zero(elt), J2=J₂)
  ℋ⃗₀ = [ℋ₁₀, ℋ₂₀]
  H⃗₀ = [MPO(elt, ℋ₀, s) for ℋ₀ in ℋ⃗₀]
  ψ₀ = complex.(MPS(elt, s, j -> isodd(j) ? "↑" : "↓"))
  ψₜ_ode = tdvp(ode_solver_f⃗, H⃗₀, time_stop, ψ₀; time_step, maxdim, cutoff, nsite)
  ψₜ_krylov = tdvp(krylov_solver_f⃗, H⃗₀, time_stop, ψ₀; time_step, cutoff, nsite)
  ψₜ_full, _ = ode_solver_f⃗(prod.(H⃗₀), time_stop, prod(ψ₀))

  @test ITensors.scalartype(ψ₀) == complex(elt)
  @test ITensors.scalartype(ψₜ_ode) == complex(elt)
  @test ITensors.scalartype(ψₜ_krylov) == complex(elt)
  @test ITensors.scalartype(ψₜ_full) == complex(elt)
  @test norm(ψ₀) ≈ 1
  @test norm(ψₜ_ode) ≈ 1
  @test norm(ψₜ_krylov) ≈ 1 rtol = √(eps(real(elt)))
  @test norm(ψₜ_full) ≈ 1

  ode_err = norm(prod(ψₜ_ode) - ψₜ_full)
  krylov_err = norm(prod(ψₜ_krylov) - ψₜ_full)

  @test krylov_err > ode_err
  @test ode_err < √(eps(real(elt))) * 10^4
  @test krylov_err < √(eps(real(elt))) * 10^5
end
end
