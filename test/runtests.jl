using Test

@testset "ITensorTDVP.jl" begin
  @testset "$filename" for filename in ["tdvp.jl"]
    println("Running $filename")
    include(filename)
  end
end

nothing
