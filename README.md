# ITensorTDVP

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mtfishman.github.io/ITensorTDVP.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mtfishman.github.io/ITensorTDVP.jl/dev)
[![Build Status](https://github.com/mtfishman/ITensorTDVP.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mtfishman/ITensorTDVP.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mtfishman/ITensorTDVP.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mtfishman/ITensorTDVP.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

| :information_source: NOTE          |
|:---------------------------|
| This package is effectively a generalization of the DMRG code in [ITensors.jl](https://github.com/ITensor/ITensors.jl), using the MPS/MPO types from that package. It provides a general MPS "solver" interface which allows us to implement other MPS/MPO optimization/solver functionality like DMRG (`ITensorTDVP.dmrg` or `eigsolve`), TDVP (`tdvp` or `exponentiate`), linear solving (`linsolve`), DMRG-X (`dmrg_x`), etc. while sharing most of the code across those different functions. Therefore, it effectively supercedes the DMRG functionality in ITensors.jl (`dmrg`), and provides its own `ITensorTDVP.dmrg`/`eigsolve` function that is essentially the same as the `dmrg` function from ITensors.jl. This package is fairly stable and appropriate for general use. The primary missing feature is a lack of modern subspace expansion tools for methods like TDVP. However, 2-site TDVP or TEBD is often sufficient for performing subspace expansion (except when [it's not](https://arxiv.org/abs/2005.06104)). |
| However, note that future developments, including modern subspace expansion tools, are being developed in our next-generation tensor network library [ITensorNetworks.jl](https://github.com/mtfishman/ITensorNetworks.jl). That package has contraction, optimization, and evolution tools for general tensor networks, as well as methods like DMRG, TDVP, and linear solving for tree tensor networks, which effectively supercedes the functionality of this package which is limited to solvers for just MPS/MPO (linear/path graph) tensor networks. ITensorNetworks.jl is under heavy development and is _not_ meant for general usage at the moment, except for those who are brave enough to handle missing features and breaking interfaces. Basically, for the average user who wants stable and reliable code, if you need to use TDVP or linear solving on MPS, you should use this package, but keep in mind that it will be superceded by ITensorNetworks.jl eventually, once that code catches up on functionality for TTN that we currently have for MPS/MPO in ITensors.jl (like Hamiltonian construction from `OpSum`, general TTN algebra like addition and contraction, etc.). If you need to use functionality like DMRG, TDVP, or linear solving for TTN, you can try out ITensorNetworks.jl, but keep in mind that it is still a work in progress. |

To install this package, you can use the following steps:
```
$ julia

julia> ]

pkg> add ITensorTDVP
```
