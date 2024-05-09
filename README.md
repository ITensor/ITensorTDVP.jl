| :warning: WARNING          |
|:---------------------------|
| The [ITensorTDVP.jl](https://github.com/ITensor/ITensorTDVP.jl) package will be deprecated in favor of the [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl) package. We plan to move all of the code from this package into ITensorMPS.jl. For now, to help with backwards compatability, ITensorMPS.jl simply re-exports the functionality of ITensorTDVP.jl. To prepare for the change, please change `using ITensorTDVP` to `using ITensorMPS` in your code. |

# ITensorTDVP

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
However, as noted above we now recommend installing and loading `ITensorMPS` instead of `ITensorTDVP`.

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
