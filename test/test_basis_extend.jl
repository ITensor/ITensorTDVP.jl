@eval module $(gensym())
using ITensors.ITensorMPS:
  OpSum, MPO, MPS, dmrg, inner, linkdims, maxlinkdim, randomMPS, siteinds
using ITensorTDVP: expand_basis, tdvp
using LinearAlgebra: normalize
using Test: @test, @testset
const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "expand_basis_orthogonalize (conserve_qns=$conserve_qns, eltype=$elt)" for conserve_qns in
                                                                                    (
    false, true
  ),
  elt in elts

  n = 6
  s = siteinds("S=1/2", n; conserve_qns)
  state = randomMPS(s, j -> isodd(j) ? "↑" : "↓"; linkdims=4)
  reference = randomMPS(s, j -> isodd(j) ? "↑" : "↓"; linkdims=2)
  state_expanded = expand_basis(state, [reference]; alg="orthogonalize")
  @test inner(state_expanded, state) ≈ inner(state, state)
  @test inner(state_expanded, reference) ≈ inner(state, reference)
end
@testset "basis_extend" begin
  n = 10
  s = siteinds("S=1/2", n)
  opsum = OpSum()
  for j in 1:(n - 1)
    opsum += 0.5, "S+", j, "S-", j + 1
    opsum += 0.5, "S-", j, "S+", j + 1
    opsum += "Sz", j, "Sz", j + 1
  end
  operator = MPO(opsum, s)
  state = MPS(s, n -> isodd(n) ? "Up" : "Dn")
  state_expanded = expand_basis(state, operator; alg="global_krylov")
  @test maxlinkdim(state_expanded) > 1
  @test inner(state_expanded, state) ≈ inner(state, state)
end
@testset "Decoupled Ladder" begin
  nx = 10
  ny = 2
  n = nx * ny
  s = siteinds("S=1/2", n)
  opsum = OpSum()
  for j in 1:2:(n - 2)
    opsum += "Sz", j, "Sz", j + 2
    opsum += 1 / 2, "S+", j, "S-", j + 2
    opsum += 1 / 2, "S-", j, "S+", j + 2
  end
  for j in 2:2:(n - 2)
    opsum += "Sz", j, "Sz", j + 2
    opsum += 1 / 2, "S+", j, "S-", j + 2
    opsum += 1 / 2, "S-", j, "S+", j + 2
  end
  operator = MPO(opsum, s)
  nexpansions = 10
  init = randomMPS(s; linkdims=2)
  reference_energy, reference_state = dmrg(
    operator,
    init;
    nsweeps=10,
    maxdim=[10, 10, 20, 20, 40, 80, 100],
    cutoff=1e-8,
    noise=1e-10,
  )
  state = randomMPS(s)
  tau = 0.5
  for step in 1:nexpansions
    state = expand_basis(state, operator; alg="global_krylov", cutoff=1e-4)
    state = tdvp(operator, -4tau, state; nsteps=4, cutoff=1e-5)
    state = normalize(state)
  end
  @test inner(state', operator, state) ≈ reference_energy rtol = 1e-3
end
end
