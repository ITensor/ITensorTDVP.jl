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

### ITensorTDVP.jl v0.4 Release Notes

#### Breaking changes

- When calling `tdvp(operator, t, init; kwargs...)`, `t` is now interpreted as the total evolution time, while in `ITensorTDVP.jl` it was being interpreted as the time step. To upgrade, change code like:
```julia
t = -0.1im
nsweeps = 10
tdvp(operator, t, init; nsweeps, kwargs...)
```
to:
```julia
t = -1.0im
nsweeps = 10
tdvp(operator, t, init; nsweeps, kwargs...)
```
to evolve to time `1.0` over `10` steps with time steps of `0.1`. Also note that `ITensorTDVP.jl` v0.4 instroduces `nsteps` as an alias for `nsweeps`, so you can also use:
```julia
t = -1.0im
nsteps = 10
tdvp(operator, t, init; nsteps, kwargs...)
```
or specify the time steps instead of the number of steps:
```julia
t = -1.0im
time_step = -0.1im
tdvp(operator, t, init; time_step, kwargs...)
```
- In `tdvp`, a custom local updater/solver must now be passed as a keyword argument `updater`, as opposed to as the first argument which was the syntax in `ITensorTDVP.jl` v0.3 and below. So code like:
```julia
tdvp(custom_updater, operator, t, init; kwargs...)
```
must be changed to:
```julia
tdvp(operator, t, init; updater=custom_updater, kwargs...)
```
- In 
- The keyword argument `psi` that was being passed to observers in `tdvp`, `linsolve`, etc. which stored the current state has been renamed to `state`. Change code like:
```julia
measure_sz(; psi) = expect(psi, "Sz")
obs = observer("Sz" => measure_sz)
tdvp(operator, t, init; (observer!)=obs, kwargs...)
```
to:
```julia
measure_sz(; state) = expect(state, "Sz")
obs = observer("Sz" => measure_sz)
tdvp(operator, t, init; (observer!)=obs, kwargs...)
```
- Only the argument ordering `tdvp(operator, t, init; kwargs...)` is now supported. `tdvp(t, operator, init; kwargs...)` and `tdvp(operator, init, t; kwargs...)` have been removed.
- In `tdvp`, the keyword argument `solver_backend` has been renamed to `updater_backend`. Change code like:
```julia
tdvp(operator, t, init; solver_backend="applyexp", kwargs...)
```
to:
```julia
tdvp(operator, t, init; updater_backend="applyexp", kwargs...)
```
- In `tdvp` and `ITensorTDVP.dmrg`, keyword arguments passed to the local solver should now be passed in a NamedTuple in the `updater_kwargs` keyword argument, such as `updater_kwargs=(; tol=1e-5, krylovdim=20)`, instead of as keyword arguments `solver_tol`, `solver_krylovdim`, etc. Change code like:
```julia
tdvp(operator, t, init; solver_tol=1e-5, krylovdim=20, kwargs...)
ITensorTDVP.dmrg(operator, init; solver_tol=1e-5, solver_krylovdim=20, kwargs...)
```
to:
```julia
tdvp(operator, t, init; updater_kwargs=(; tol=1e-5, krylovdim=20), kwargs...)
ITensorTDVP.dmrg(operator, init; updater_kwargs=(; tol=1e-5, krylovdim=20), kwargs...)
```
- In `linsolve`, the keyword argument `solver_kwargs` has been renamed to `updater_kwargs`.
- In `ITensorTDVP.dmrg`, `dmrg_x`, and `linsolve`, the keyword argument `step_observer!` has been renamed to `sweep_observer!`. Either name is allowed in `tdvp`.
- Support for `ITensors.AbstractObserver`-based observers has been removed, use `Observers.observer` instead.
- In `contract(operator::MPO, state::MPS; alg="fit", kwargs...)`, and `apply(operator::MPO, state::MPS; kwargs...)`, the keyword argument for specifying an initial guess for the result is now called `init` instead of `init_mps`. Additionally, in `contract`, `init` should have primed site indices, or more generally should have site indices which are those that are not shared by the input operator and state.
- In custom local updaters/solvers, the keyword arguments `time_step`, `current_time`, and `outputlevel` are now being passed as a NamedTuple in a new keyword argument `internal_kwargs`. Change local updaters/solvers from:
```julia
function custom_updater(operator, init; time_step, current_time, outputlevel, kwargs...)
  ## updater code
end
```
to:
```julia
function custom_updater(operator, init; internal_kwargs, kwargs...)
  (; time_step, current_time, outputlevel) = internal_kwargs
  # Or if the updater only needs to use a subset of them:
  # (; time_step) = internal_kwargs
  ## updater code
end
```

#### New features

- `nsteps` is now an alias for the `nsweeps` keyword argument in `tdvp`.
- `TimeDependentSum` now accepts coefficients and terms that are Tuples, along with the previous interface which accepted Vectors.
- Custom local updaters/solvers can be passed in a keyword argument `updater` to `ITensorTDVP.dmrg`, `dmrg_x`, and `linsolve`, which is consistent with the new syntax for `tdvp`.

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
