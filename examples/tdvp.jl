using ITensors
using ITensorTDVP

n = 10
s = siteinds("S=1/2", n)

function heisenberg(n)
  os = OpSum()
  for j in 1:(n - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  return os
end

H = MPO(heisenberg(n), s)
ψ = randomMPS(s, "↑"; linkdims=10)

@show inner(ψ', H, ψ)

_, ϕ = tdvp(H, ψ; t=-0.1*im, exponentiate_krylovdim=30, nsweeps=10, maxdim=50, cutoff=1e-10)

e2, ϕ2 = dmrg(H, ψ; nsweeps=10, maxdim=20, cutoff=1e-10)

@show inner(ϕ', H, ϕ)
