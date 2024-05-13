using KrylovKit: eigsolve, exponentiate

default_nsweeps() = nothing
default_checkdone() = Returns(false)
default_write_when_maxdim_exceeds() = nothing
default_nsite() = 2
default_reverse_step() = false
default_time_start() = nothing
default_time_step() = nothing
default_order() = 2
default_observer() = EmptyObserver()
default_sweep_observer() = EmptyObserver()
default_outputlevel() = 0
default_normalize() = false
default_sweep() = 1
default_current_time() = 0

# Truncation
default_maxdim() = typemax(Int)
default_mindim() = 1
default_cutoff(type::Type{<:Number}) = eps(real(type))
default_noise() = 0

# Solvers
default_tdvp_updater_backend() = "exponentiate"
default_ishermitian() = true
default_issymmetric() = true
default_updater_verbosity() = 0

# Customizable based on updater function
default_updater_outputlevel(::Function) = 0

default_updater_tol(::Function) = error("Not implemented")
default_updater_which_eigenvalue(::Function) = error("Not implemented")
default_updater_krylovdim(::Function) = error("Not implemented")
default_updater_verbosity(::Function) = error("Not implemented")

## Solver-specific keyword argument defaults

# dmrg/eigsolve
default_updater_tol(::typeof(eigsolve)) = 1e-14
default_updater_krylovdim(::typeof(eigsolve)) = 3
default_updater_maxiter(::typeof(eigsolve)) = 1
default_updater_which_eigenvalue(::typeof(eigsolve)) = :SR

# tdvp/exponentiate
default_updater_tol(::typeof(exponentiate)) = 1e-12
default_updater_krylovdim(::typeof(exponentiate)) = 30
default_updater_maxiter(::typeof(exponentiate)) = 100

# tdvp/applyexp
default_updater_tol(::typeof(applyexp)) = 1e-12
default_updater_krylovdim(::typeof(applyexp)) = 30
default_updater_maxiter(::typeof(applyexp)) = 100
