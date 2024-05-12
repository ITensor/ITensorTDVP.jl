using ITensorMPS: MPO, OpSum, dmrg, inner, randomMPS, siteinds, tdvp

function main()
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

  @show inner(ψ', H, ψ) / inner(ψ, ψ)

  ϕ = tdvp(
    H,
    -20.0,
    ψ;
    time_step=-1.0,
    reverse_step=false,
    normalize=true,
    maxdim=30,
    cutoff=1e-10,
    outputlevel=1,
  )
  @show inner(ϕ', H, ϕ) / inner(ϕ, ϕ)

  e2, ϕ2 = dmrg(H, ψ; nsweeps=10, maxdim=20, cutoff=1e-10)
  @show inner(ϕ2', H, ϕ2) / inner(ϕ2, ϕ2), e2

  return nothing
end

main()
