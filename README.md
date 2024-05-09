| :warning: WARNING          |
|:---------------------------|
| The [ITensorTDVP.jl](https://github.com/ITensor/ITensorTDVP.jl) package will be deprecated in favor of the [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl) package. We plan to move all of the code from this package into ITensorMPS.jl. For now, to help with backwards compatability, ITensorMPS.jl simply re-exports the functionality of ITensorTDVP.jl. To prepare for the change, please change `using ITensorTDVP` to `using ITensorMPS` in your code. |

# ITensorTDVP

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mtfishman.github.io/ITensorTDVP.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mtfishman.github.io/ITensorTDVP.jl/dev)
[![Build Status](https://github.com/mtfishman/ITensorTDVP.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mtfishman/ITensorTDVP.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mtfishman/ITensorTDVP.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mtfishman/ITensorTDVP.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

## Installation

To install this package, you can use the following steps:
```
$ julia

julia> ]

pkg> add ITensorTDVP
```

## News

### ITensorTDVP.jl v0.3 Release Notes

#### Breaking changes

- `ITensorTDVP.dmrg` and `ITensorTDVP.dmrg_x` now output a tuple containing the eigenvalue and eigenvector, while before they just output the eigenvector. You should update code like this:
```julia
psi = dmrg_x(H, psi0; nsweeps=10, maxdim=100, cutoff=1e-6)
psi = ITensorTDVP.dmrg(H, psi0; nsweeps=10, maxdim=100, cutoff=1e-6)
```
to:
```julia
energy, psi = dmrg_x(H, psi0; nsweeps=10, maxdim=100, cutoff=1e-6)
energy, psi = ITensorTDVP.dmrg(H, psi0; nsweeps=10, maxdim=100, cutoff=1e-6)
```

### ITensorTDVP.jl v0.2 Release Notes

#### Breaking changes

- ITensorTDVP.jl v0.2.0-v0.2.4: The `applyexp` Krylov exponentiation solver backend was removed, and `solver_backend="applyexp"` option for `tdvp` now just calls `exponentiate` from KrylovKit.jl. `applyexp` is in many ways the same as `exponentiate` bit `exponentiate` has more advanced features like restarts. In these versions, `solver_backend="applyexp"` prints a warning to that effect. As of ITensorTDVP.jl v0.2.5, we have brought back the `applyexp` backend because we received reports that it performed better in certain cases. We plan to investigate that issue and make sure `exponentiate` works as well as `applyexp` in those cases so that we can go back to just having a single `exponentiate` backend.

#### Bug fixes

- `svd_alg` now doesn't specify a default value, so the default value is set by the `svd` function in ITensors/NDTensors. This fixes an issue using ITensorTDVP.jl and GPU backends, where the default value being set in ITensorTDVP.jl wasn't compatible with the options available in some GPU backends like CUDA.
- More generally, keyword arguments are handled better throughout the package, so default values are handled more systematically and keyword arguments are listed or forwarded more explicitly, so it should catch more mistakes like passing an incorrect keyword argument name.

## About

This package is effectively a generalization of the DMRG code in [ITensors.jl](https://github.com/ITensor/ITensors.jl), using the MPS/MPO types from that package. It provides a general MPS "solver" interface which allows us to implement a variety of MPS/MPO optimization/solver functionality like DMRG (`ITensorTDVP.dmrg`), TDVP (`ITensorTDVP.tdvp`), linear solving (`ITensorTDVP.linsolve`/`KrylovKit.linsolve`), DMRG-X (`ITensorTDVP.dmrg_x`), etc. while sharing most of the code across those different functions. Therefore, it effectively supercedes the DMRG functionality in ITensors.jl (`dmrg`), and provides its own `ITensorTDVP.dmrg` function that is essentially the same as the `dmrg` function from ITensors.jl (though for now it only outputs the state, while `ITensors.dmrg` outputs the energy and the state, likely we will make the interface more similar to `ITensors.dmrg` in future versions of the code). This package is fairly stable and appropriate for general use. The primary missing feature is a lack of modern subspace expansion tools for methods like TDVP and 1-site DMRG. However, 2-site TDVP or TEBD is often sufficient for performing subspace expansion (except when [it's not](https://arxiv.org/abs/2005.06104)).

However, note that future developments, including modern subspace expansion tools, are being developed in our next-generation tensor network library [ITensorNetworks.jl](https://github.com/mtfishman/ITensorNetworks.jl). The goal of that package is to provide contraction, optimization, and evolution tools for general tensor networks, as well as methods like DMRG, TDVP, and linear solving for tree tensor networks, and the eventual goal is to replace this package which is limited to solvers for just MPS/MPO (linear/path graph) tensor networks. However, ITensorNetworks.jl is under heavy development and is _not_ meant for general usage at the moment, except for those who are brave enough to handle missing features and breaking interfaces. Basically, for the average user who wants stable and reliable code, if you need to use MPS-based TDVP or linear solving, you should use this package for the time being.
