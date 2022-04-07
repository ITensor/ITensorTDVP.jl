
struct TDVPInfo
  maxtruncerr::Float64
end

struct TDVPOrder
  order::Int
  direction::Base.Ordering #Forward, Reverse
end

function tdvp(solver,PH,time_step::Number,order::TDVPOrder,psi0::MPS;kwargs...)
  psi=copy(psi0)
  return tdvp!(solver,PH,time_step,order,psi;kwargs...)
end

function tdvp!(solver,PH,time_step::Number,tdvp_order::TDVPOrder,psi::MPS;kwargs...)
  order=tdvp_order.order
  direction=tdvp_order.direction
  
  if order==1
    sub_time_steps=[1.0,0.0]
    orderings=[direction, Base.ReverseOrdering(direction)]
  elseif order==2
    sub_time_steps=[1.0/2.0,1.0/2.0]
    orderings=[direction, Base.ReverseOrdering(direction)]
  elseif order==4
    s=1.0/(2.0-2.0^(1.0/3.0))
    sub_time_steps=[s/2.0,s/2.0,(1.0-2.0*s)/2.,(1.0-2.0*s)/2.0,s/2.0,s/2.0]
    orderings=repeat([direction, Base.ReverseOrdering(direction)],3)
  end
  
  sub_time_steps*=time_step
  global info
  for substep in 1:length(sub_time_steps)
    psi, PH, info=tdvp!(solver, PH, sub_time_steps[substep], orderings[substep], psi; kwargs...)
  end
  return psi, PH, info
  end

function tdvp(solver, PH, time_step::Number, direction::Base.Ordering, psi0::MPS; kwargs...)
  psi=copy(psi0)
  return tdvp!(solver, PH, time_step, direction, psi; kwargs...)
end

function tdvp!(solver, PH, time_step::Number, direction::Base.Ordering, psi::MPS; kwargs...)
  if length(psi) == 1
    error(
      "`tdvp` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.exponentiate`, etc.",
    )
  end

  nsite::Int = get(kwargs, :nsite, 2)
  reverse_step::Bool = get(kwargs, :reverse_step, true)
  normalize::Bool = get(kwargs, :normalize, false)
  which_decomp::Union{String,Nothing} = get(kwargs, :which_decomp, nothing)
  svd_alg::String = get(kwargs, :svd_alg, "divide_and_conquer")
  observer = get(kwargs, :observer, NoObserver())
  outputlevel = get(kwargs, :outputlevel, 0)
  sw = get(kwargs, :sweep, 1)

  maxdim::Integer = get(kwargs, :maxdim, typemax(Int))
  mindim::Integer = get(kwargs, :mindim, 1)
  cutoff::Real = get(kwargs, :cutoff, 1E-16)
  noise::Real = get(kwargs, :noise, 0.0)

  N = length(psi)
  set_nsite!(PH, nsite)
  if direction==Base.Forward
    if !isortho(psi) || orthocenter(psi) != 1
      orthogonalize!(psi, 1)
    end
    @assert isortho(psi) && orthocenter(psi) == 1
    position!(PH, psi, 1)
  elseif direction==Base.Reverse
    if !isortho(psi) || orthocenter(psi) != N-nsite+1
      orthogonalize!(psi, N-nsite+1)
    end
    @assert(isortho(psi) && (orthocenter(psi) == N-nsite+1))
    position!(PH, psi, N-nsite+1)
  end

  maxtruncerr = 0.0
  for (b, ha) in sweepnext(N; ncenter=nsite)
    # unidirectional (half-)sweeps only, skip over the other direction
    if direction==Base.Forward && ha==2
      continue
    elseif direction==Base.Reverse && ha==1
      continue
    end
    
    # Do 'forwards' evolution step
    set_nsite!(PH, nsite)
    position!(PH, psi, b)
    if nsite == 1
      phi1 = psi[b]
    elseif nsite == 2
      phi1 = psi[b] * psi[b + 1]
    end
    phi1, info = solver(PH, time_step, phi1)

    normalize && (phi1 /= norm(phi1))

    spec = nothing
    if nsite == 1
      psi[b] = phi1
    elseif nsite == 2
      ortho = ha == 1 ? "left" : "right"

      drho = nothing
      if noise > 0.0 && ha == 1
        drho = noise * noiseterm(PH, phi, ortho)
      end

      spec = replacebond!(
        psi,
        b,
        phi1;
        maxdim,
        mindim,
        cutoff,
        eigen_perturbation=drho,
        ortho=ortho,
        normalize,
        which_decomp,
        svd_alg,
      )
      maxtruncerr = max(maxtruncerr, spec.truncerr)
    end

    #
    # Do backwards evolution step
    #
    if reverse_step && (ha == 1 && (b + nsite - 1 != N)) || (ha == 2 && b != 1)
      b1 = (ha == 1 ? b + 1 : b)
      Δ = (ha == 1 ? +1 : -1)
      if nsite == 2
        phi0 = psi[b1]
      elseif nsite == 1
        uinds = uniqueinds(phi1, psi[b + Δ])
        U, S, V = svd(phi1, uinds)
        psi[b] = U
        phi0 = S * V
        if ha==1
          ITensors.setleftlim!(psi,b)
        elseif ha==2
          ITensors.setrightlim!(psi,b)
        end
      end
      
      set_nsite!(PH, nsite-1)
      position!(PH, psi, b1)

      phi0, info = solver(PH, -time_step, phi0)

      normalize && (phi0 ./= norm(phi0))

      if nsite == 2
        psi[b1] = phi0
      elseif nsite == 1
        psi[b + Δ] = phi0 * psi[b + Δ]
        if ha==1
           ITensors.setrightlim!(psi,b + Δ + 1)
        elseif ha==2
           ITensors.setleftlim!(psi,b + Δ - 1)
        end
      end
      set_nsite!(PH, nsite)
    end

    if outputlevel >= 2
      @printf("Sweep %d, half %d, bond (%d,%d) \n", sw, ha, b, b + 1)
      @printf(
        "  Truncated using cutoff=%.1E maxdim=%d mindim=%d\n", cutoff, maxdim, mindim,
      )
      if spec != nothing
        @printf(
          "  Trunc. err=%.2E, bond dimension %d\n", spec.truncerr, dim(linkind(psi, b))
        )
      end
      flush(stdout)
    end

    half_sweep_is_done = ((b == 1 && ha == 2) || (b == N && ha == 1) )
    if observer isa Observers.Observer
      update!(
        observer; psi, bond=b, sweep=sw, half_sweep=ha, spec, outputlevel, half_sweep_is_done
      )
    elseif observer isa ITensors.AbstractObserver
      measure!(
        observer; psi, bond=b, sweep=sw, half_sweep=ha, spec, outputlevel, half_sweep_is_done
      )
    else
      error("observer has unrecognized type ($(typeof(observer)))")
    end
  end

  # Just to be sure:
  normalize && normalize!(psi)

  return psi, PH, TDVPInfo(maxtruncerr)
end

function exponentiate_solver(; kwargs...)
  solver_kwargs = (;
    ishermitian=get(kwargs, :ishermitian, true),
    issymmetric=get(kwargs, :issymmetric, true),
    tol=get(kwargs, :solver_tol, 1E-12),
    krylovdim=get(kwargs, :solver_krylovdim, 30),
    maxiter=get(kwargs, :solver_maxiter, 100),
    verbosity=get(kwargs, :solver_outputlevel, 0),
    eager=true,
  )
  function solver(H, t, psi0; kws...)
    psi, info = exponentiate(H, t, psi0; solver_kwargs..., kws...)
    return psi, info
  end
  return solver
end

function applyexp_solver(; kwargs...)
  tol_per_unit_time = get(kwargs, :solver_tol, 1E-8)
  solver_kwargs = (;
    maxiter=get(kwargs, :solver_krylovdim, 30),
    outputlevel=get(kwargs, :solver_outputlevel, 0),
  )
  function solver(H, t, psi0; kws...)
    #apply_exp tol is absolute, compute from tol_per_unit_time:
    tol = abs(t) * tol_per_unit_time
    psi, info = apply_exp(H, t, psi0; tol, solver_kwargs..., kws...)
    return psi, info
  end
  return solver
end

function tdvp_solver(; kwargs...)
  solver_backend = get(kwargs, :solver_backend, "exponentiate")
  if solver_backend == "applyexp"
    return applyexp_solver(; kwargs...)
  elseif solver_backend == "exponentiate"
    return exponentiate_solver(; kwargs...)
  else
    error(
      "solver_backend=$solver_backend not recognized (options are \"applyexp\" or \"exponentiate\")",
    )
  end
end

function eigsolve_solver(; kwargs...)
  howmany = 1
  which = get(kwargs, :solver_which_eigenvalue, :SR)
  solver_kwargs = (;
    ishermitian=get(kwargs, :ishermitian, true),
    tol=get(kwargs, :solver_tol, 1E-14),
    krylovdim=get(kwargs, :solver_krylovdim, 3),
    maxiter=get(kwargs, :solver_maxiter, 1),
    verbosity=get(kwargs, :solver_verbosity, 0),
  )
  function solver(H, t, psi0; kws...)
    vals, vecs, info = eigsolve(H, psi0, howmany, which; solver_kwargs..., kws...)
    psi = vecs[1]
    return psi, info
  end
  return solver
end

function _tdvp_compute_nsweeps(t; kwargs...)
  time_step::Number = get(kwargs, :time_step, t)
  nsweeps::Integer = get(kwargs, :nsweeps, 0)
  if nsweeps > 0 && time_step != t
    error("Cannot specify both time_step and nsweeps in tdvp")
  elseif isfinite(time_step) && abs(time_step) > 0.0 && nsweeps == 0
    nsweeps = convert(Int, ceil(abs(t / time_step)))
    if !(nsweeps * time_step ≈ t)
      error("Time step $time_step not commensurate with total time t=$t")
    end
  end

  return nsweeps
end

function _extend_sweeps_param(param, nsweeps)
  if param isa Number
    eparam = fill(param, nsweeps)
  else
    length(param) == nsweeps && return param
    eparam = Vector(undef, nsweeps)
    eparam[1:length(param)] = param
    eparam[(length(param) + 1):end] .= param[end]
  end
  return eparam
end

function process_sweeps(; kwargs...)
  nsweeps = get(kwargs, :nsweeps, 1)
  maxdim = get(kwargs, :maxdim, fill(typemax(Int), nsweeps))
  mindim = get(kwargs, :mindim, fill(1, nsweeps))
  cutoff = get(kwargs, :cutoff, fill(1E-16, nsweeps))
  noise = get(kwargs, :noise, fill(0.0, nsweeps))

  maxdim = _extend_sweeps_param(maxdim, nsweeps)
  mindim = _extend_sweeps_param(mindim, nsweeps)
  cutoff = _extend_sweeps_param(cutoff, nsweeps)
  noise = _extend_sweeps_param(noise, nsweeps)

  return (; maxdim, mindim, cutoff, noise)
end

function tdvp(solver, PH, t::Number, psi0::MPS; kwargs...)
  reverse_step = true

  nsweeps = _tdvp_compute_nsweeps(t; kwargs...)
  maxdim, mindim, cutoff, noise = process_sweeps(; nsweeps, kwargs...)

  time_step::Number = get(kwargs, :time_step, t)
  order = get(kwargs, :order, 2)
  tdvp_order = TDVPOrder(order,Base.Forward)

  checkdone = get(kwargs, :checkdone, nothing)
  write_when_maxdim_exceeds::Union{Int,Nothing} = get(
    kwargs, :write_when_maxdim_exceeds, nothing
  )
  observer = get(kwargs, :observer, NoObserver())
  outputlevel::Int = get(kwargs, :outputlevel, 0)

  psi = copy(psi0)

  for sw in 1:nsweeps
    if !isnothing(write_when_maxdim_exceeds) && maxdim[sw] > write_when_maxdim_exceeds
      if outputlevel >= 2
        println(
          "write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim(sweeps, sw) = $(maxdim(sweeps, sw)), writing environment tensors to disk",
        )
      end
      PH = disk(PH)
    end

    sw_time = @elapsed begin
      psi, PH, info = tdvp!(
        solver,
        PH,
        time_step,
        tdvp_order,
        psi;
        kwargs...,
        reverse_step,
        sweep=sw,
        maxdim=maxdim[sw],
        mindim=mindim[sw],
        cutoff=cutoff[sw],
        noise=noise[sw],
      )
    end

    if outputlevel >= 1
      @printf(
        "After sweep %d maxlinkdim=%d maxerr=%.2E time=%.3f\n",
        sw,
        maxlinkdim(psi),
        info.maxtruncerr,
        sw_time
      )
      flush(stdout)
    end

    isdone = false
    if checkdone != nothing
      isdone = checkdone(; psi, sweep=sw, outputlevel, kwargs...)
    elseif observer isa ITensors.AbstractObserver
      isdone = checkdone!(observer; psi, sweep=sw, outputlevel)
    end
    isdone && break
  end
  return psi
end

function tdvp(H, t::Number, psi0::MPS; kwargs...)
  return tdvp(tdvp_solver(; kwargs...), H, t, psi0, sweeps; kwargs...)
end

function dmrg(H, psi0::MPS; kwargs...)
  t = Inf # DMRG is TDVP with an infinite timestep and no reverse step
  reverse_step = false
  psi = tdvp(eigsolve_solver(; kwargs...), H, t, psi0; reverse_step, kwargs...)
  return psi
end

function dmrg(H::MPO, psi0::MPS; kwargs...)
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  H = ITensors.permute(H, (linkind, siteinds, linkind))
  PH = ProjMPO(H)
  return dmrg(PH, psi0; kwargs...)
end

"""
    tdvp(Hs::Vector{MPO},psi0::MPS,t::Number; kwargs...)
    tdvp(Hs::Vector{MPO},psi0::MPS,t::Number, sweeps::Sweeps; kwargs...)

Use the time dependent variational principle (TDVP) algorithm
to compute `exp(t*H)*psi0` using an efficient algorithm based
on alternating optimization of the MPS tensors and local Krylov
exponentiation of H.
                    
This version of `tdvp` accepts a representation of H as a
Vector of MPOs, Hs = [H1,H2,H3,...] such that H is defined
as H = H1+H2+H3+...
Note that this sum of MPOs is not actually computed; rather
the set of MPOs [H1,H2,H3,..] is efficiently looped over at 
each step of the algorithm when optimizing the MPS.

Returns:
* `psi::MPS` - time-evolved MPS
"""
function tdvp(solver, Hs::Vector{MPO}, t::Number, psi0::MPS; kwargs...)
  for H in Hs
    check_hascommoninds(siteinds, H, psi0)
    check_hascommoninds(siteinds, H, psi0')
  end
  Hs .= ITensors.permute.(Hs, Ref((linkind, siteinds, linkind)))
  PHs = ProjMPOSum(Hs)
  return tdvp(solver, PHs, t, psi0; kwargs...)
end

function tdvp(H::Vector{MPO}, t::Number, psi0::MPS; kwargs...)
  return tdvp(tdvp_solver(; kwargs...), H, t, psi0; kwargs...)
end

"""
    tdvp(H::MPO,psi0::MPS,t::Number; kwargs...)
    tdvp(H::MPO,psi0::MPS,t::Number; kwargs...)

Use the time dependent variational principle (TDVP) algorithm
to compute `exp(t*H)*psi0` using an efficient algorithm based
on alternating optimization of the MPS tensors and local Krylov
exponentiation of H.
                    
Returns:
* `psi::MPS` - time-evolved MPS

Optional keyword arguments:
* `outputlevel::Int = 1` - larger outputlevel values resulting in printing more information and 0 means no output
* `observer` - object implementing the [Observer](@ref observer) interface which can perform measurements and stop early
* `write_when_maxdim_exceeds::Int` - when the allowed maxdim exceeds this value, begin saving tensors to disk to free memory in large calculations
"""
function tdvp(solver, H::MPO, t::Number, psi0::MPS; kwargs...)
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  H = ITensors.permute(H, (linkind, siteinds, linkind))
  PH = ProjMPO(H)
  return tdvp(solver, PH, t, psi0; kwargs...)
end

function tdvp(H::MPO, t::Number, psi0::MPS; kwargs...)
  return tdvp(tdvp_solver(; kwargs...), H, t, psi0; kwargs...)
end

#
# Support for passing Sweeps objects to tdvp and dmrg
#

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
