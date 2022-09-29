using ITensors
using ITensorTDVP
using Random
using Test

@testset "Contract MPO" begin
  N = 20
  s = siteinds("S=1/2", N)
  psi = randomMPS(s; linkdims=8)

  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  for j in 1:(N - 2)
    os += 0.5, "S+", j, "S-", j + 2
    os += 0.5, "S-", j, "S+", j + 2
    os += "Sz", j, "Sz", j + 2
  end
  H = MPO(os, s)

  Hpsi = apply(H, psi; alg="fit")
  @test inner(psi, Hpsi) ≈ inner(psi', H, psi) atol = 1E-5

  # Test with nsweeps=2
  Hpsi = apply(H, psi; alg="fit", nsweeps=2)
  @test inner(psi, Hpsi) ≈ inner(psi', H, psi) atol = 1E-5

  # Test with less good initial guess MPS not equal to psi
  psi_guess = copy(psi)
  truncate!(psi_guess; maxdim=2)
  Hpsi = apply(H, psi; alg="fit", nsweeps=4, init_mps=psi_guess)
  @test inner(psi, Hpsi) ≈ inner(psi', H, psi) atol = 1E-5

  # Test with nsite=1
  Hpsi_guess = apply(H, psi; alg="naive", cutoff=1E-4)
  Hpsi = apply(H, psi; alg="fit", init_mps=Hpsi_guess, nsite=1, nsweeps=2)
  @test inner(psi, Hpsi) ≈ inner(psi', H, psi) atol = 1E-4
end

nothing
