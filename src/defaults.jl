default_nsweeps() = nothing
default_checkdone() = nothing
default_write_when_maxdim_exceeds() = nothing
default_nsite() = 2
default_reverse_step() = true
default_time_start() = 0
default_time_step(t) = t
default_order() = 2
default_observer!() = NoObserver()
default_step_observer!() = NoObserver()
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
default_tdvp_solver_backend() = "exponentiate"
default_ishermitian() = true
default_solver_verbosity() = 0

default_solver_tol(::Function) = error("Not implemented")
default_solver_which_eigenvalue(::Function) = error("Not implemented")
default_solver_krylovdim(::Function) = error("Not implemented")
default_solver_maxiter(::Function) = error("Not implemented")
default_solver_verbosity(::Function) = error("Not implemented")

## Solver-specific keyword argument defaults

# dmrg/eigsolve
default_solver_tol(::typeof(eigsolve)) = 1e-14
default_solver_which_eigenvalue(::typeof(eigsolve)) = :SR
default_solver_krylovdim(::typeof(eigsolve)) = 3
default_solver_maxiter(::typeof(eigsolve)) = 1
