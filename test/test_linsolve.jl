using ITensors
using ITensorTDVP
using Test

@testset "Linsolve" begin
  cutoff = 1E-10
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

  #x = linsolve(H, b; cutoff, nsweeps=3, ishermitian=true, outputlevel=2)
  x0 = copy(b)
  x = ITensorTDVP.dmrg_linsolve(H, b, b; cutoff, nsweeps=4, ishermitian=true)

  @show linkdims(x)
  @show norm(apply(H, x) - b)
  @show norm(apply(H, x_c) - b)

  #@show inner(x,x)
  #@show inner(x_c,x)/sqrt(inner(x,x))

end

nothing
