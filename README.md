# ITensorTDVP

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mtfishman.github.io/ITensorTDVP.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mtfishman.github.io/ITensorTDVP.jl/dev)
[![Build Status](https://github.com/mtfishman/ITensorTDVP.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mtfishman/ITensorTDVP.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mtfishman/ITensorTDVP.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mtfishman/ITensorTDVP.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

This particular branch contains an experimental draft of sweeping algorithms for tree tensor
networks, which in turn relies on several unregistered packages and experimental forks.

To install this version of the package, you can use the following steps:
```
$ julia

julia> ]

pkg> add https://github.com/mtfishman/MultiDimDictionaries.jl

pkg> add https://github.com/mtfishman/NamedGraphs.jl

pkg> add https://github.com/mtfishman/DataGraphs.jl

pkg> add https://github.com/leburgel/ITensorNetworks.jl#tree_sweeping

pkg> add https://github.com/leburgel/ITensorTDVP.jl#tree_sweeping
```
