using ITensors
using ITensorTDVP
using KrylovKit
using Printf
using Observers
using Random
using Test

@testset "Basic TDVP" begin
  N = 10
  cutoff = 1e-12

  s = siteinds("S=1/2", N)

  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end

  H = MPO(os, s)

  ψ0 = randomMPS(s; linkdims=10)

  # Time evolve forward:
  ψ1 = tdvp(H, -0.1im, ψ0; nsweeps=1, cutoff, nsite=1)

  @test norm(ψ1) ≈ 1.0

  ## Should lose fidelity:
  #@test abs(inner(ψ0,ψ1)) < 0.9

  # Average energy should be conserved:
  @test real(inner(ψ1', H, ψ1)) ≈ inner(ψ0', H, ψ0)

  # Time evolve backwards:
  ψ2 = tdvp(H, +0.1im, ψ1; nsweeps=1, cutoff)

  @test norm(ψ2) ≈ 1.0

  # Should rotate back to original state:
  @test abs(inner(ψ0, ψ2)) > 0.99
end

@testset "TDVP: Sum of Hamiltonians" begin
  N = 10
  cutoff = 1e-10

  s = siteinds("S=1/2", N)

  os1 = OpSum()
  for j in 1:(N - 1)
    os1 += 0.5, "S+", j, "S-", j + 1
    os1 += 0.5, "S-", j, "S+", j + 1
  end
  os2 = OpSum()
  for j in 1:(N - 1)
    os2 += "Sz", j, "Sz", j + 1
  end

  H1 = MPO(os1, s)
  H2 = MPO(os2, s)
  Hs = [H1, H2]

  ψ0 = randomMPS(s; linkdims=10)

  ψ1 = tdvp(Hs, -0.1im, ψ0; nsweeps=1, cutoff, nsite=1)

  @test norm(ψ1) ≈ 1.0

  ## Should lose fidelity:
  #@test abs(inner(ψ0,ψ1)) < 0.9

  # Average energy should be conserved:
  @test real(sum(H -> inner(ψ1', H, ψ1), Hs)) ≈ sum(H -> inner(ψ0', H, ψ0), Hs)

  # Time evolve backwards:
  ψ2 = tdvp(Hs,+0.1im, ψ1;nsweeps=1, cutoff)

  @test norm(ψ2) ≈ 1.0

  # Should rotate back to original state:
  @test abs(inner(ψ0, ψ2)) > 0.99
end

@testset "Custom solver in TDVP" begin
  N = 10
  cutoff = 1e-12

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
    solver_kwargs = (;
      ishermitian=true, tol=1e-12, krylovdim=30, maxiter=100, verbosity=0, eager=true
    )
    psi, info = exponentiate(PH, t, psi0; solver_kwargs...)
    return psi, info
  end

  ψ1 = tdvp(solver, H, -0.1im, ψ0; cutoff, nsite=1)

  @test norm(ψ1) ≈ 1.0

  ## Should lose fidelity:
  #@test abs(inner(ψ0,ψ1)) < 0.9

  # Average energy should be conserved:
  @test real(inner(ψ1', H, ψ1)) ≈ inner(ψ0', H, ψ0)

  # Time evolve backwards:
  ψ2 = tdvp(H, +0.1im, ψ1; cutoff)

  @test norm(ψ2) ≈ 1.0

  # Should rotate back to original state:
  @test abs(inner(ψ0, ψ2)) > 0.99
end

@testset "Accuracy Test" begin
  N = 4
  tau = 0.1
  ttotal = 1.0
  cutoff = 1e-12

  s = siteinds("S=1/2", N; conserve_qns=false)

  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  H = MPO(os, s)
  HM = prod(H)

  Ut = exp(-im * tau * HM)

  psi = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
  psix = prod(psi)

  Sz_tdvp = Float64[]
  Sz_exact = Float64[]

  c = div(N, 2)
  Szc = op("Sz", s[c])

  Nsteps = Int(ttotal / tau)
  for step in 1:Nsteps
    psix = noprime(Ut * psix)
    psix /= norm(psix)

    psi = tdvp(
      H,
      -im * tau,
      psi;
      cutoff,
      
      normalize=false,
      solver_tol=1e-12,
      solver_maxiter=500,
      solver_krylovdim=25,
    )
    push!(Sz_tdvp, real(expect(psi, "Sz"; sites=c:c)[1]))
    push!(Sz_exact, real(scalar(dag(prime(psix, s[c])) * Szc * psix)))
    F = abs(scalar(dag(psix) * prod(psi)))
    #@printf("%s=%.10f  exact=%.10f  F=%.10f\n",method,Sz_tdvp[end],Sz_exact[end],F)
  end
  
  @test norm(Sz_tdvp - Sz_exact) < 1e-5
end

@testset "TEBD Comparison" begin
  N = 10
  cutoff = 1e-12
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
  phi = copy(psi)
  c = div(N, 2)

  #
  # Evolve using TEBD
  # 

  Nsteps = convert(Int, ceil(abs(ttotal / tau)))
  Sz1 = zeros(Nsteps)
  En1 = zeros(Nsteps)
  Sz2 = zeros(Nsteps)
  En2 = zeros(Nsteps)
  

  for step in 1:Nsteps
    psi = apply(gates, psi; cutoff)
    #normalize!(psi)

    nsite = (step <= 3 ? 2 : 1)
    phi = tdvp(H,  -tau * im, phi; nsweeps=1, cutoff, nsite, normalize=true, exponentiate_krylovdim=15)

    Sz1[step] = expect(psi, "Sz"; sites=c:c)[1]
    Sz2[step] = expect(phi, "Sz"; sites=c:c)[1]
    En1[step] = real(inner(psi', H, psi))
    En2[step] = real(inner(phi', H, phi))
  end

  #
  # Evolve using TDVP
  # 
  struct TDVPObserver <: AbstractObserver end

  Sz2 = zeros(Nsteps)
  En2 = zeros(Nsteps)
  function ITensors.measure!(obs::TDVPObserver; sweep, bond, half_sweep, psi, kwargs...)
    if bond == 1 && half_sweep == 2
      Sz2[sweep] = expect(psi, "Sz"; sites=c)
      En2[sweep] = real(inner(psi', H, psi))
      #@printf("sweep %d Sz=%.12f energy=%.12f\n",sweep,Sz2[sweep],En2[sweep])
    end
  end

  phi = productMPS(s, n -> isodd(n) ? "Up" : "Dn")

  phi = tdvp(
    H,
    -im * ttotal,
    phi;
    time_step=-im * tau,
    cutoff,
    normalize=false,
    observer=TDVPObserver(),
  )

  #display(En1)
  #display(En2)
  #display(Sz1)
  #display(Sz2)
  #@show norm(Sz1 - Sz2)
  #@show norm(En1 - En2)

  @test norm(Sz1 - Sz2) < 1e-3
  @test norm(En1 - En2) < 1e-3
end

@testset "Imaginary Time Evolution" begin
  N = 10
  cutoff = 1e-12
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

  psi = randomMPS(s; linkdims=2)

  trange = 0.0:tau:ttotal
  for (step, t) in enumerate(trange)
    nsite = (step <= 10 ? 2 : 1)
    psi = tdvp(H, -tau,psi; cutoff, nsite, normalize=true, exponentiate_krylovdim=15)
    @printf("%.3f energy = %.12f\n", step * tau, inner(psi', H, psi))
  end
  #@show maxlinkdim(psi)

  @test inner(psi', H, psi) < -4.25
end

@testset "DMRG" begin
  N = 10
  cutoff = 1e-12

  s = siteinds("S=1/2", N)

  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end

  H = MPO(os, s)

  psi = randomMPS(s; linkdims=2)

  nsweeps = 10
  nsite = 2
  maxdim = [10, 20, 40, 100]
  sweeps = Sweeps(nsweeps) # number of sweeps is 5
  maxdim!(sweeps,10,20,40,100) # gradually increase states kept
  cutoff!(sweeps,cutoff)
  psi = ITensorTDVP.dmrg(
    H, psi; nsweeps, maxdim, cutoff, nsite, solver_krylovdim=3, solver_maxiter=1
  )

  e2, psi2 = dmrg(H, psi, sweeps; normalize=false, outputlevel=0)

  @test inner(psi', H, psi) ≈ inner(psi2', H, psi2)
end

@testset "Observers" begin
  N = 10
  cutoff = 1e-12
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

  c = div(N, 2)

  #
  # Using the ITensors observer system
  # 
  struct TDVPObserver <: AbstractObserver end

  Nsteps = convert(Int, ceil(abs(ttotal / tau)))
  Sz1 = zeros(Nsteps)
  En1 = zeros(Nsteps)
  function ITensors.measure!(obs::TDVPObserver; sweep, bond, half_sweep, psi, kwargs...)
    if bond == 1 && half_sweep == 2
      Sz1[sweep] = expect(psi, "Sz"; sites=c)
      En1[sweep] = real(inner(psi', H, psi))
      #@printf("sweep %d Sz=%.12f energy=%.12f\n",sweep,Sz2[sweep],En2[sweep])
    end
  end

  psi1 = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
  tdvp(
    H,
    -im * ttotal,
    psi1;
    time_step=-im * tau,
    cutoff,
    normalize=false,
    observer=TDVPObserver(),
  )

  #
  # Using Observers.jl
  # 

  function measure_sz(; psi, bond, half_sweep)
    if bond == 1 && half_sweep == 2
      return expect(psi, "Sz"; sites=c)
    end
    return nothing
  end

  function measure_en(; psi, bond, half_sweep)
    if bond == 1 && half_sweep == 2
      return real(inner(psi', H, psi))
    end
    return nothing
  end

  obs = Observer("Sz" => measure_sz, "En" => measure_en)

  psi2 = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
  tdvp(H, -im * ttotal, psi2; time_step=-im * tau, cutoff, normalize=false, observer=obs)

  # Using filter here just due to the current
  # behavior of Observers that nothing gets appended:
  Sz2 = results(obs)["Sz"]
  En2 = results(obs)["En"]

  #display(En1)
  #display(En2)
  #display(Sz1)
  #display(Sz2)
  #@show norm(Sz1 - Sz2)
  #@show norm(En1 - En2)

  @test Sz1 ≈ Sz2
  @test En1 ≈ En2
end

@testset "DMRG-X" begin
  function heisenberg(n; h=zeros(n))
    os = OpSum()
    for j in 1:(n - 1)
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
      os += "Sz", j, "Sz", j + 1
    end
    for j in 1:n
      if h[j] ≠ 0
        os -= h[j], "Sz", j
      end
    end
    return os
  end

  n = 10
  s = siteinds("S=1/2", n)

  Random.seed!(12)

  W = 12
  # Random fields h ∈ [-W, W]
  h = W * (2 * rand(n) .- 1)
  H = MPO(heisenberg(n; h), s)

  initstate = rand(["↑", "↓"], n)
  ψ = MPS(s, initstate)

  dmrg_x_kwargs = (
    nsweeps=20, reverse_step=false, normalize=true, maxdim=20, cutoff=1e-10, outputlevel=0
  )

  ϕ = dmrg_x(ProjMPO(H), ψ; dmrg_x_kwargs...)

  @test inner(ψ', H, ψ) / inner(ψ, ψ) ≈ inner(ϕ', H, ϕ) / inner(ϕ, ϕ) rtol = 1e-1
  @test inner(H, ψ, H, ψ) ≉ inner(ψ', H, ψ)^2 atol = 1e-1
  @test inner(H, ϕ, H, ϕ) ≈ inner(ϕ', H, ϕ)^2 atol = 1e-7
end

nothing
