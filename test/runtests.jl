using Test
using ITensorTDVP

test_path = joinpath(pkgdir(ITensorTDVP), "test")
test_files = filter(
  file -> startswith(file, "test_") && endswith(file, ".jl"), readdir(test_path; join=true)
)
@testset "ITensorTDVP.jl" begin
  @testset "$filename" for filename in testfiles
    println("Running $filename")
    include(filename)
  end
end

nothing
