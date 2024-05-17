@eval module $(gensym())
using ITensors: ITensors, MPO, OpSum, inner, randomMPS, siteinds
using ITensorTDVP: ITensorTDVP
using StableRNGs: StableRNG
using Test: @test, @test_throws, @testset
@testset "DMRG (eltype=$elt, nsite=$nsite, conserve_qns=$conserve_qns)" for elt in (
    Float32, Float64, Complex{Float32}, Complex{Float64}
  ),
  nsite in [1, 2],
  conserve_qns in [false, true]

  N = 10
  cutoff = eps(real(elt)) * 10^4
  s = siteinds("S=1/2", N; conserve_qns)
  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  H = MPO(elt, os, s)
  rng = StableRNG(1234)
  psi = randomMPS(rng, elt, s, j -> isodd(j) ? "↑" : "↓"; linkdims=20)
  nsweeps = 10
  maxdim = [10, 20, 40, 100]
  @test_throws ErrorException ITensorTDVP.dmrg(H, psi; maxdim, cutoff, nsite)
  e, psi = ITensorTDVP.dmrg(
    H, psi; nsweeps, maxdim, cutoff, nsite, updater_kwargs=(; krylovdim=3, maxiter=1)
  )
  @test inner(psi', H, psi) ≈ e
  e2, psi2 = ITensors.dmrg(H, psi; nsweeps, maxdim, cutoff, outputlevel=0)
  @test ITensors.scalartype(psi2) == elt
  @test e2 isa real(elt)
  @test e ≈ e2 rtol = √(eps(real(elt))) * 10
  @test inner(psi', H, psi) ≈ inner(psi2', H, psi2) rtol = √(eps(real(elt))) * 10
end
end
