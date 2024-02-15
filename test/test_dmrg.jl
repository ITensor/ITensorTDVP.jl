using ITensors
using ITensorTDVP
using Random
using Test

@testset "DMRG (eltype=$elt, nsite=$nsite)" for elt in (
    Float32, Float64, Complex{Float32}, Complex{Float64}
  ),
  nsite in [1, 2]

  N = 10
  cutoff = eps(real(elt)) * 10^4
  s = siteinds("S=1/2", N)
  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  H = MPO(elt, os, s)
  psi = randomMPS(elt, s; linkdims=20)
  nsweeps = 10
  maxdim = [10, 20, 40, 100]
  sweeps = Sweeps(nsweeps) # number of sweeps is 5
  maxdim!(sweeps, 10, 20, 40, 100) # gradually increase states kept
  cutoff!(sweeps, cutoff)
  psi = ITensorTDVP.dmrg(
    H, psi; nsweeps, maxdim, cutoff, nsite, solver_krylovdim=3, solver_maxiter=1
  )
  e2, psi2 = dmrg(H, psi, sweeps; outputlevel=0)
  @test ITensors.scalartype(psi2) == elt
  @test e2 isa real(elt)
  @test inner(psi', H, psi) ≈ inner(psi2', H, psi2) rtol = √(eps(real(elt))) * 10
end

nothing
