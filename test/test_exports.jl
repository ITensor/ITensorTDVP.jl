@eval module $(gensym())
using ITensorTDVP: ITensorTDVP
using Test: @test, @testset
@testset "Test exports" begin
  @test issetequal(
    names(ITensorTDVP),
    [:ITensorTDVP, :TimeDependentSum, :dmrg_x, :expand_basis, :linsolve, :tdvp, :to_vec],
  )
end
end
