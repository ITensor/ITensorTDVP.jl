using Test
using ITensorTDVP

test_path = joinpath(pkgdir(ITensorTDVP), "test")
test_files = filter(
  file -> startswith(file, "test_") && endswith(file, ".jl"), readdir(test_path)
)
@testset "ITensorTDVP.jl" begin
  @testset "$filename" for filename in test_files
    println("Running $filename")
    include(joinpath(test_path, filename))
  end
end

nothing
