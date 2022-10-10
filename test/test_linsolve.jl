using ITensors
using ITensorTDVP
using Test

@testset "Linsolve" begin
  cutoff = 1E-11
  maxdim = 8
  nsweeps = 2

  N = 8
  s = siteinds("S=1/2", N)

  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  H = MPO(os, s)

  # Correct x is x_c
  x_c = randomMPS(s; linkdims=4)
  # Compute b
  b = apply(H, x_c; cutoff)

  x0 = randomMPS(s;linkdims=10)
  x = linsolve(H, b, x0; cutoff, maxdim, nsweeps, ishermitian=true, solver_tol=1E-14)

  @show linkdims(x)
  @show norm(apply(H, x) - b)
  @show norm(x-x_c)

end

nothing
