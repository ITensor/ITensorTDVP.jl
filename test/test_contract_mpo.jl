@eval module $(gensym())
using ITensors: ITensors, MPO, OpSum, apply, delta, inner, randomMPS, siteinds, truncate!
using ITensorTDVP: ITensorTDVP
using Test: @test, @testset
@testset "Contract MPO (eltype=$elt)" for elt in (
  Float32, Float64, Complex{Float32}, Complex{Float64}
)
  N = 20
  s = siteinds("S=1/2", N)
  psi = randomMPS(elt, s; linkdims=8)
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
  H = MPO(elt, os, s)

  # Test basic usage with default parameters
  Hpsi = apply(H, psi; alg="fit")
  @test ITensors.scalartype(Hpsi) == elt
  @test inner(psi, Hpsi) ≈ inner(psi', H, psi) atol = 1e-5

  #
  # Change "top" indices of MPO to be a different set
  #
  t = siteinds("S=1/2", N)
  psit = deepcopy(psi)
  for j in 1:N
    H[j] *= delta(elt, s[j]', t[j])
    psit[j] *= delta(elt, s[j], t[j])
  end

  # Test with nsweeps=2
  Hpsi = apply(H, psi; alg="fit", nsweeps=2)
  @test ITensors.scalartype(Hpsi) == elt
  @test inner(psit, Hpsi) ≈ inner(psit, H, psi) atol = 1e-5

  # Test with less good initial guess MPS not equal to psi
  psi_guess = copy(psi)
  truncate!(psi_guess; maxdim=2)
  Hpsi = apply(H, psi; alg="fit", nsweeps=4, init_mps=psi_guess)
  @test ITensors.scalartype(Hpsi) == elt
  @test inner(psit, Hpsi) ≈ inner(psit, H, psi) atol = 1e-5

  # Test with nsite=1
  Hpsi_guess = apply(H, psi; alg="naive", cutoff=1E-4)
  Hpsi = apply(H, psi; alg="fit", init_mps=Hpsi_guess, nsite=1, nsweeps=2)
  @test ITensors.scalartype(Hpsi) == elt
  @test inner(psit, Hpsi) ≈ inner(psit, H, psi) atol = 1e-4
end
end
