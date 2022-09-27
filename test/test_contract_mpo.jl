using ITensors
using ITensorTDVP
using Random
using Test

@testset "Contract MPO" begin
  N = 20
  s = siteinds("S=1/2", N)
  psi = randomMPS(s; linkdims=10)

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

  Hpsi = fit_contract_mpo(H, psi)
  @test inner(psi, Hpsi) ≈ inner(psi', H, psi) atol = 1E-5

  Hpsi = fit_contract_mpo(H, psi; nsweeps=2)
  @test inner(psi, Hpsi) ≈ inner(psi', H, psi) atol = 1E-6
end

nothing
