using Test
using ITensors
using ITensorTDVP
using Printf


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

@testset "Custom solver in TDVP" begin
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

  function solver(PH, t, psi0)
    solver_kwargs = (; ishermitian=true,
                     tol=1e-12,
                     krylovdim=30,
                     maxiter=100,
                     verbosity=0)
    psi, info = exponentiate(PH, t, psi0; solver_kwargs...)
    return psi, info
  end

  ψ1 = tdvp(solver, H, ψ0, -0.1im; cutoff, nsite=1)

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

@testset "Accuracy Test" begin

  N = 4
  tau = 0.02
  ttotal = 1.0
  cutoff = 1E-12
  use_trotter = false

  method = use_trotter ? "tebd" : "tdvp"

  s = siteinds("S=1/2", N; conserve_qns=false)

  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  H = MPO(os, s)
  HM = prod(H)

  if use_trotter
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
  end

  Ut = exp(-im*tau*HM)

  psi = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
  psix = prod(psi)

  Sz_tdvp = Float64[]
  Sz_exact = Float64[]

  c = div(N,2)
  Szc = op("Sz",s[c])
  Nsteps = Int(ttotal / tau)
  for step in 1:Nsteps
    psix = noprime(Ut*psix)
    psix /= norm(psix)

    if use_trotter
      psi = apply(gates,psi; cutoff)
      normalize!(psi)
    else
      psi = tdvp(H,psi,-tau*im; cutoff, 
                                normalize=true,
                                exponentiate_tol=1E-12,
                                exponentiate_maxiter=500,
                                exponentiate_krylovdim=100)
    end


    push!(Sz_tdvp, real(expect(psi, "Sz"; site_range=c:c)))
    push!(Sz_exact, real(scalar(dag(prime(psix,s[c]))*Szc*psix)))
    #F = abs(scalar(dag(psix)*prod(psi)))
    #@printf("%s=%.10f  exact=%.10f  F=%.10f\n",method,Sz_tdvp[end],Sz_exact[end],F)
  end

  #@show norm(Sz_tdvp-Sz_exact)
  @test norm(Sz_tdvp-Sz_exact) < 1E-5

end


@testset "TEBD Comparison" begin
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
    phi = tdvp(H,phi,-tau*im; cutoff, 
                              nsite, 
                              normalize=true,
                              exponentiate_krylovdim=15)

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

@testset "Imaginary Time Evolution" begin
  N = 10
  cutoff = 1E-12
  tau = 1.0
  ttotal = 50.0

  s = siteinds("S=1/2", N)

  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end

  H = MPO(os, s)

  psi = randomMPS(s;linkdims=2)

  Nsteps = Int(ttotal / tau)
  for step in 1:Nsteps
    nsite = (step <= 10 ? 2 : 1)
    psi = tdvp(H,psi,-tau; cutoff, 
                           nsite, 
                           normalize=true,
                           exponentiate_krylovdim=15)
    @printf("%.3f energy = %.12f\n",step*tau,inner(psi,H,psi))
  end
  #@show maxlinkdim(psi)

  @test inner(psi,H,psi) < -4.25

end

nothing
