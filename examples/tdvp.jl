using ITensors
using ITensorTDVP
n = 25
s = siteinds("S=1/2", n,conserve_qns=true)

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
ψ = randomMPS(s,i -> isodd(i) ? "Up" : "Dn"; linkdims=10)

E0= inner(ψ', H, ψ) / inner(ψ, ψ)


ϕ = tdvp(
  H,
  ψ,
  -0.1im;
  nsweeps=20,
  nsite=1,
  reverse_step=true,
  normalize=false,
  maxdim=50,
  cutoff=5e-2,
  atol=1e-11,
  outputlevel=1,
)
@show inner(ϕ', H, ϕ) / inner(ϕ, ϕ) , E0

ϕ3 = ITensors.dmrg(H, ψ; nsweeps=10, maxdim=50, cutoff=1e-10)

@show inner(ϕ3', H, ϕ3) / inner(ϕ3, ϕ3)
