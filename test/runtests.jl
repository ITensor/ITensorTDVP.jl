using ITensors
using ITensorTDVP
using Test
using Printf


@testset "ITensorTDVP.jl" begin

@testset "Basic TDVP" begin
  N = 10
  cutoff = 1E-10

  s = siteinds("S=1/2", N)

  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end

  H = MPO(os, s)

  ψ0 = randomMPS(s; linkdims=10)

  ψ1 = tdvp(H, ψ0, -0.1im; cutoff, nsite=1)

  @test norm(ψ1) ≈ 1.0

  ## Should lose fidelity:
  #@test abs(inner(ψ0,ψ1)) < 0.9

  # Average energy should be conserved:
  @test real(inner(ψ1,H,ψ1)) ≈ inner(ψ0,H,ψ0)

  # Time evolve backwards:
  ψ2 = tdvp(H, ψ1, +0.1im; cutoff)

  @test norm(ψ2) ≈ 1.0

  # Should rotate back to original state:
  @test abs(inner(ψ0,ψ2)) > 0.99
end


@testset "TDVP Comparison" begin
  N = 10
  cutoff = 1E-12
  tau = 0.1
  ttotal = 1.0

  s = siteinds("S=1/2", N; conserve_qns=true)

  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end

  H = MPO(os, s)

  gates = ITensor[]
  for j in 1:(N - 1)
    s1 = s[j]
    s2 = s[j + 1]
    hj =
      op("Sz", s1) * op("Sz", s2) +
      1 / 2 * op("S+", s1) * op("S-", s2) +
      1 / 2 * op("S-", s1) * op("S+", s2)
    Gj = exp(-1.0im * tau / 2 * hj)
    push!(gates, Gj)
  end
  append!(gates, reverse(gates))

  psi = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
  phi = productMPS(s, n -> isodd(n) ? "Up" : "Dn")

  c = div(N, 2) # center site

  Nsteps = Int(ttotal / tau)
  Sz1 = zeros(Nsteps)
  Sz2 = zeros(Nsteps)
  En1 = zeros(Nsteps)
  En2 = zeros(Nsteps)

  for step in 1:Nsteps
    psi = apply(gates, psi; cutoff)
    normalize!(psi)

    nsite = (step <= 3 ? 2 : 1)
    phi = tdvp(H,phi,-tau*im; cutoff, nsite, exponentiate_krylovdim=15)

    Sz1[step] = expect(psi, "Sz"; site_range=c:c)
    Sz2[step] = expect(phi, "Sz"; site_range=c:c)
    En1[step] = real(inner(psi,H,psi))
    En2[step] = real(inner(phi,H,phi))
  end

  #display(En1)
  #display(En2)
  #display(Sz1)
  #display(Sz2)

  @test norm(Sz1-Sz2) < 1E-3
  @test norm(En1-En2) < 1E-3

end


end

nothing
