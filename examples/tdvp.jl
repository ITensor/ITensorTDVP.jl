using ITensors
using ITensorTDVP
n = 10
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
  0.2im;
  nsweeps=1,
  nsite=1,
  reverse_step=true,
  normalize=true,
  maxdim=50,
  cutoff=1e-10,
  outputlevel=1,
)
ϕ = tdvp(
  H,
  ϕ,
  0.2im;
  nsweeps=1,
  nsite=1,
  reverse_step=true,
  normalize=true,
  maxdim=50,
  cutoff=1e-10,
  outputlevel=1,
)
ϕ = tdvp(
  H,
  ϕ,
  0.2im;
  nsweeps=2,
  nsite=1,
  reverse_step=true,
  normalize=true,
  maxdim=100,
  cutoff=1e-10,
  outputlevel=1,
)
@show inner(ϕ', H, ϕ) / inner(ϕ, ϕ) , E0

e2, ϕ2 = dmrg(H, ψ; nsweeps=10, maxdim=20, cutoff=1e-10)

@show inner(ϕ2', H, ϕ2) / inner(ϕ2, ϕ2)

ϕ3 = ITensorTDVP.dmrg(H, ψ; nsweeps=10, maxdim=20, cutoff=1e-10)

@show inner(ϕ3', H, ϕ3) / inner(ϕ3, ϕ3)
