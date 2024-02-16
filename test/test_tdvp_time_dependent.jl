@eval module $(gensym())
using ITensors:
  Index, MPO, ProjMPO, ProjMPOSum, QN, randomITensor, randomMPS, position!, siteinds
using ITensorTDVP: ITensorTDVP, TimeDependentSum, to_vec
using Test: @test, @test_skip, @testset
@testset "TDVP with ODE local solver" begin
  @testset "to_vec (eltype=$elt)" for elt in (
      Float32, Float64, Complex{Float32}, Complex{Float64}
    ),
    space in (2, [QN(0) => 1, QN(1) => 1])

    i, j = Index.((space, space))
    a = randomITensor(elt, i, j)
    v, to_itensor = to_vec(a)
    @test v isa Vector{elt}
    @test vec(Array(a, i, j)) == v
    @test to_itensor(v) == a
  end
  @testset "TimeDependentSum (eltype=$elt)" for elt in (
      Float32, Float64, Complex{Float32}, Complex{Float64}
    ),
    conserve_qns in [false, true]

    n = 4
    s = siteinds("S=1/2", 4; conserve_qns)
    H = MPO(elt, s, "I")
    H⃗ = [H, H]
    region = 2:3
    ψ = randomMPS(elt, s, j -> isodd(j) ? "↑" : "↓"; linkdims=2)
    H⃗ᵣ = ProjMPO.(H⃗)
    map(Hᵣ -> position!(Hᵣ, ψ, first(region)), H⃗ᵣ)
    ∑Hᵣ = ProjMPOSum(H⃗)
    position!(∑Hᵣ, ψ, first(region))
    f⃗ₜ = [t -> sin(elt(0.1) * t), t -> cos(elt(0.2) * t)]
    α = elt(0.5)
    ∑Hₜ = α * TimeDependentSum(f⃗ₜ, ∑Hᵣ)
    t₀ = elt(0.5)
    ∑Hₜ₀ = ∑Hₜ(t₀)
    ψᵣ = reduce(*, map(v -> ψ[v], region))
    Hψ = ∑Hₜ₀(ψᵣ)
    @test eltype(Hψ) == elt
    @test Hψ ≈ sum(i -> α * f⃗ₜ[i](t₀) * H⃗ᵣ[i](ψᵣ), eachindex(H⃗))
  end
  @testset "Time dependent TDVP" begin
    # These tests take too long to compile, skip for now.
    @test_skip include("tdvp_time_dependent.jl")
  end
end
end
