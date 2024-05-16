@eval module $(gensym())
using ITensors: scalartype
using ITensors.ITensorMPS: OpSum, MPO, MPS, inner, linkdims, maxlinkdim, randomMPS, siteinds
using ITensorTDVP: dmrg, expand, tdvp
using LinearAlgebra: normalize
using Random: Random
using Test: @test, @testset
const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "expand (eltype=$elt)" for elt in elts
  @testset "expand (alg=\"orthogonalize\", conserve_qns=$conserve_qns, eltype=$elt)" for conserve_qns in
                                                                                         (
    false, true
  )
    n = 6
    s = siteinds("S=1/2", n; conserve_qns)
    Random.seed!(1234)
    state = randomMPS(elt, s, j -> isodd(j) ? "↑" : "↓"; linkdims=4)
    reference = randomMPS(elt, s, j -> isodd(j) ? "↑" : "↓"; linkdims=2)
    state_expanded = expand(state, [reference]; alg="orthogonalize")
    @test scalartype(state_expanded) === elt
    @test inner(state_expanded, state) ≈ inner(state, state)
    @test inner(state_expanded, reference) ≈ inner(state, reference)
  end
  @testset "expand (alg=\"global_krylov\", conserve_qns=$conserve_qns, eltype=$elt)" for conserve_qns in
                                                                                         (
    false, true
  )
    n = 10
    s = siteinds("S=1/2", n; conserve_qns)
    opsum = OpSum()
    for j in 1:(n - 1)
      opsum += 0.5, "S+", j, "S-", j + 1
      opsum += 0.5, "S-", j, "S+", j + 1
      opsum += "Sz", j, "Sz", j + 1
    end
    operator = MPO(elt, opsum, s)
    state = MPS(elt, s, j -> isodd(j) ? "↑" : "↓")
    state_expanded = expand(state, operator; alg="global_krylov")
    @test scalartype(state_expanded) === elt
    @test maxlinkdim(state_expanded) > 1
    @test inner(state_expanded, state) ≈ inner(state, state)
  end
  @testset "Decoupled ladder (alg=\"global_krylov\", eltype=$elt)" begin
    nx = 10
    ny = 2
    n = nx * ny
    s = siteinds("S=1/2", n)
    opsum = OpSum()
    for j in 1:2:(n - 2)
      opsum += 1 / 2, "S+", j, "S-", j + 2
      opsum += 1 / 2, "S-", j, "S+", j + 2
      opsum += "Sz", j, "Sz", j + 2
    end
    for j in 2:2:(n - 2)
      opsum += 1 / 2, "S+", j, "S-", j + 2
      opsum += 1 / 2, "S-", j, "S+", j + 2
      opsum += "Sz", j, "Sz", j + 2
    end
    operator = MPO(elt, opsum, s)
    Random.seed!(1234)
    init = randomMPS(elt, s; linkdims=30)
    reference_energy, reference_state = dmrg(
      operator,
      init;
      nsweeps=15,
      maxdim=[10, 10, 20, 20, 40, 80, 100],
      cutoff=(√(eps(real(elt)))),
      noise=(√(eps(real(elt)))),
    )
    Random.seed!(1234)
    state = randomMPS(elt, s)
    nexpansions = 10
    tau = elt(0.5)
    for step in 1:nexpansions
      # TODO: Use `∜` instead of `fourthroot` in Julia 1.10 and above.
      state = expand(
        state, operator; alg="global_krylov", krylovdim=3, cutoff=fourthroot(eps(real(elt)))
      )
      state = tdvp(
        operator,
        -4tau,
        state;
        nsteps=4,
        cutoff=1e-5,
        updater_kwargs=(; tol=1e-3, krylovdim=5),
      )
      state = normalize(state)
    end
    @test scalartype(state) === elt
    # TODO: Use `∜` instead of `fourthroot` in Julia 1.10 and above.
    @test inner(state', operator, state) ≈ reference_energy rtol =
      2 * fourthroot(eps(real(elt)))
  end
end
end
