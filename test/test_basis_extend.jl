using ITensors
import ITensors: pause
using ITensorTDVP
using Random
using Test
using Printf

@testset "extend function" begin
  N = 6
  s = siteinds(2,N)

  psi = randomMPS(s;linkdims=4)
  phi = randomMPS(s;linkdims=2)

  psix = ITensorTDVP.extend(psi,[phi])
  @test inner(psix,psi) ≈ inner(psi,psi)
  @test inner(psix,phi) ≈ inner(psi,phi)
end

@testset "basis_extend" begin
  N = 10

  s = siteinds("S=1/2", N)

  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  H = MPO(os, s)

  ψ0 = productMPS(s,n->isodd(n) ? "Up" : "Dn")
  ψx = basis_extend(ψ0,H)
  @test maxlinkdim(ψx) > 1
  @test inner(ψx,ψ0) ≈ inner(ψ0,ψ0)
end

@testset "Decoupled Ladder" begin
  cutoff = 1E-5
  Nx = 10
  Ny = 2
  N = Nx*Ny
  s = siteinds("S=1/2",N)

  hterms = OpSum()
  for j=1:2:(N-2)
    hterms += "Sz",j,"Sz",j+2
    hterms += 1/2,"S+",j,"S-",j+2
    hterms += 1/2,"S-",j,"S+",j+2
  end
  for j=2:2:(N-2)
    hterms += "Sz",j,"Sz",j+2
    hterms += 1/2,"S+",j,"S-",j+2
    hterms += 1/2,"S-",j,"S+",j+2
  end
  H = MPO(hterms,s)

  tau = -0.5
  Nstep = 40


  ψ0 = randomMPS(s;linkdims=2)
  energy,ψg = dmrg(H,ψ0; nsweeps=10,noise=1E-10,maxdim=[10,10,20,20,40,80,100],cutoff=1E-8)
  @show energy
  pause()

  ψ = randomMPS(s;linkdims=1)
  @show maxlinkdim(ψ)
  @show inner(ψ',H,ψ)
  for n=1:Nstep
    @show maxlinkdim(ψ)
    if n%4==1
      ψ = basis_extend(ψ,H)
      @show linkdims(ψ)
    end
    ψ = tdvp(H,tau,ψ; cutoff)
    normalize!(ψ)
    @show maxlinkdim(ψ)
    @show inner(ψ',H,ψ)
    println()
  end


end


nothing
