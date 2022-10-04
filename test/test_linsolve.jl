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

  # Correct x
  x_c = randomMPS(s; linkdims=4)

  b = apply(H, x_c; cutoff)

  #x0 = copy(b)
  #x0 = copy(x_c)
  x0 = randomMPS(s;linkdims=10)
  x = linsolve(H, b, x0; cutoff, maxdim, nsweeps, ishermitian=true)

  @show linkdims(x)
  @show norm(apply(H, x) - b)
  @show norm(apply(H, x_c) - b)
  @show norm(x-x_c)

  #@show inner(x,x)
  #@show inner(x_c,x)/sqrt(inner(x,x))

end

nothing
