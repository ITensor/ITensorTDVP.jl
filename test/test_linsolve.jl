using ITensors
using ITensorTDVP
using Test
using Random

@testset "linsolve with conserve_qns=$conserve_qns and eltype=$eltype" for conserve_qns in
                                                                           (false, true),
  eltype in (Float32, Float64, ComplexF32, ComplexF64)

  N = 6
  s = siteinds("S=1/2", N; conserve_qns)

  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  H = ITensors.convert_leaf_eltype(eltype, MPO(os, s))

  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]

  Random.seed!(1234)
  x_c = randomMPS(eltype, s, state; linkdims=2)
  e, x_c = dmrg(H, x_c; nsweeps=10, cutoff=1e-6, maxdim=20, outputlevel=0)

  @test ITensors.scalartype(x_c) == eltype

  # Compute `b = H * x_c`
  b = apply(H, x_c; cutoff=1e-8)

  @test ITensors.scalartype(b) == eltype

  # Starting guess
  x0 = x_c + eltype(0.05) * randomMPS(eltype, s, state; linkdims=2)

  @test ITensors.scalartype(x0) == eltype

  nsweeps = 10
  cutoff = 1e-5
  maxdim = 20
  solver_kwargs = (; tol=1e-4, maxiter=20, krylovdim=30, ishermitian=true)
  x = @time linsolve(H, b, x0; nsweeps, cutoff, maxdim, solver_kwargs)

  @test ITensors.scalartype(x) == eltype

  @show norm(x - x_c)
  @test norm(x - x_c) < 1e-2
end

nothing
