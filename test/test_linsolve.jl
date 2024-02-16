@eval module $(gensym())
using ITensors: ITensors, MPO, OpSum, apply, randomMPS, siteinds
using ITensorTDVP: ITensorTDVP, dmrg
using KrylovKit: linsolve
using LinearAlgebra: norm
using Test: @test, @testset
using Random: Random
@testset "linsolve (eltype=$elt, conserve_qns=$conserve_qns)" for elt in (
    Float32, Float64, Complex{Float32}, Complex{Float64}
  ),
  conserve_qns in [false, true]

  N = 6
  s = siteinds("S=1/2", N; conserve_qns)
  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  H = ITensors.convert_leaf_eltype(elt, MPO(os, s))
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  Random.seed!(1234)
  x_c = randomMPS(elt, s, state; linkdims=2)
  x_c = dmrg(H, x_c; nsweeps=10, cutoff=1e-6, maxdim=20, outputlevel=0)
  @test ITensors.scalartype(x_c) == elt
  # Compute `b = H * x_c`
  b = apply(H, x_c; cutoff=1e-8)
  @test ITensors.scalartype(b) == elt
  # Starting guess
  x0 = x_c + elt(0.05) * randomMPS(elt, s, state; linkdims=2)
  @test ITensors.scalartype(x0) == elt
  nsweeps = 10
  cutoff = 1e-5
  maxdim = 20
  solver_kwargs = (; tol=1e-4, maxiter=20, krylovdim=30, ishermitian=true)
  x = linsolve(H, b, x0; nsweeps, cutoff, maxdim, solver_kwargs)
  @test ITensors.scalartype(x) == elt
  @test norm(x - x_c) < 1e-2
end
end
