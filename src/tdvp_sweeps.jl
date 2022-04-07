function process_sweeps(s::Sweeps)
  return (;
    nsweeps=s.nsweep, maxdim=s.maxdim, mindim=s.mindim, cutoff=s.cutoff, noise=s.noise
  )
end

function tdvp(X, t::Number, psi0::MPS, sweeps::Sweeps; kwargs...)
  return tdvp(X, t, psi0; process_sweeps(sweeps)..., kwargs...)
end

function tdvp(X1, X2, t::Number, psi0::MPS, sweeps::Sweeps; kwargs...)
  return tdvp(X1, X2, t, psi0; process_sweeps(sweeps)..., kwargs...)
end

function dmrg(X, psi0::MPS, sweeps::Sweeps; kwargs...)
  return dmrg(X, psi0; process_sweeps(sweeps)..., kwargs...)
end

function dmrg(X1, X2, psi0::MPS, sweeps::Sweeps; kwargs...)
  return dmrg(X1, X2, psi0; process_sweeps(sweeps)..., kwargs...)
end
