using ITensors
using ITensorTDVP
using Test

@testset "ITensorTDVP.jl" begin

@testset "Basic TDVP" begin
  N = 10
  s = siteinds("S=1/2", N)

  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end

  H = MPO(os, s)

  ψ0 = randomMPS(s, "↑"; linkdims=10)

  ψ1 = tdvp(H, ψ0, -0.1im; nsweeps=1, maxdim=50, cutoff=1e-10)

  @test norm(ψ1) ≈ 1.0

  # Should lose fidelity:
  @test abs(inner(ψ0,ψ1)) < 0.5

  # Average energy should be conserved:
  @test real(inner(ψ1,H,ψ1)) ≈ inner(ψ0,H,ψ0)

  # Time evolve backwards:
  ψ2 = tdvp(H, ψ1, +0.1im; nsweeps=1, maxdim=50, cutoff=1e-10)

  @test norm(ψ2) ≈ 1.0

  # Should rotate back to original state:
  @test abs(inner(ψ0,ψ2)) ≈ 1.0
end


end

nothing
