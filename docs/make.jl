using ITensorTDVP
using Documenter

DocMeta.setdocmeta!(ITensorTDVP, :DocTestSetup, :(using ITensorTDVP); recursive=true)

makedocs(;
  modules=[ITensorTDVP],
  authors="Matthew Fishman <mfishman@flatironinstitute.org> and contributors",
  repo="https://github.com/mtfishman/ITensorTDVP.jl/blob/{commit}{path}#{line}",
  sitename="ITensorTDVP.jl",
  format=Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://mtfishman.github.io/ITensorTDVP.jl",
    assets=String[],
  ),
  pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/mtfishman/ITensorTDVP.jl", devbranch="main")
