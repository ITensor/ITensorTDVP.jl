@eval module $(gensym())
using ITensors: ITensors, contract
using ITensors.ITensorMPS: MPO, MPS, siteinds
using ITensorTDVP: ITensorTDVP, tdvp
using LinearAlgebra: norm
using Test: @test, @testset

include(joinpath(pkgdir(ITensorTDVP), "examples", "03_models.jl"))
include(joinpath(pkgdir(ITensorTDVP), "examples", "03_updaters.jl"))

@testset "Time dependent Hamiltonian (eltype=$elt, conserve_qns=$conserve_qns)" for elt in
                                                                                    (
    Float32, Float64, Complex{Float32}, Complex{Float64}
  ),
  conserve_qns in [false, true]

  n = 4
  J₁ = elt(1)
  J₂ = elt(0.1)
  ω₁ = real(elt)(0.1)
  ω₂ = real(elt)(0.2)
  # TODO: Make into a Tuple.
  ω⃗ = [ω₁, ω₂]
  f⃗ = map(ω -> (t -> cos(ω * t)), ω⃗)
  time_step = real(elt)(0.1)
  time_stop = real(elt)(1)
  nsite = 2
  maxdim = 100
  cutoff = √(eps(real(elt)))
  tol = √eps(real(elt))
  s = siteinds("S=1/2", n)
  ℋ₁₀ = heisenberg(n; J=J₁, J2=zero(elt))
  ℋ₂₀ = heisenberg(n; J=zero(elt), J2=J₂)
  # TODO: Make into a Tuple.
  ℋ⃗₀ = [ℋ₁₀, ℋ₂₀]
  # TODO: Make into a Tuple.
  H⃗₀ = [MPO(elt, ℋ₀, s) for ℋ₀ in ℋ⃗₀]
  ψ₀ = complex.(MPS(elt, s, j -> isodd(j) ? "↑" : "↓"))
  ψₜ_ode = tdvp(
    -im * TimeDependentSum(f⃗, H⃗₀),
    time_stop,
    ψ₀;
    updater=ode_updater,
    updater_kwargs=(; reltol=tol, abstol=tol),
    time_step,
    maxdim,
    cutoff,
    nsite,
  )
  ψₜ_krylov = tdvp(
    -im * TimeDependentSum(f⃗, H⃗₀),
    time_stop,
    ψ₀;
    updater=krylov_updater,
    updater_kwargs=(; tol, eager=true),
    time_step,
    maxdim,
    cutoff,
    nsite,
  )
  ψₜ_full, _ = ode_updater(
    -im * TimeDependentSum(f⃗, contract.(H⃗₀)),
    contract(ψ₀);
    internal_kwargs=(; time_step=time_stop),
    reltol=tol,
    abstol=tol,
  )

  @test ITensors.scalartype(ψ₀) == complex(elt)
  @test ITensors.scalartype(ψₜ_ode) == complex(elt)
  @test ITensors.scalartype(ψₜ_krylov) == complex(elt)
  @test ITensors.scalartype(ψₜ_full) == complex(elt)
  @test norm(ψ₀) ≈ 1
  @test norm(ψₜ_ode) ≈ 1
  @test norm(ψₜ_krylov) ≈ 1 rtol = √(eps(real(elt)))
  @test norm(ψₜ_full) ≈ 1

  ode_err = norm(contract(ψₜ_ode) - ψₜ_full)
  krylov_err = norm(contract(ψₜ_krylov) - ψₜ_full)

  @test krylov_err > ode_err
  @test ode_err < √(eps(real(elt))) * 10^4
  @test krylov_err < √(eps(real(elt))) * 10^5
end
end
