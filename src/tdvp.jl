
struct TDVPInfo
  maxtruncerr::Float64
end

function tdvp_iteration(solver, 
                        PH, 
                        time_step::Number, 
                        psi0::MPS;
                        kwargs...)
  if length(psi0) == 1
    error(
      "`tdvp` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.exponentiate`, etc.",
    )
  end

  nsite::Int = get(kwargs, :nsite, 2)
  reverse_step::Bool = get(kwargs, :reverse_step, true)
  normalize::Bool = get(kwargs, :normalize, false)
  which_decomp::Union{String,Nothing} = get(kwargs, :which_decomp, nothing)
  svd_alg::String = get(kwargs, :svd_alg, "divide_and_conquer")
  obs = get(kwargs, :observer, NoObserver())
  outputlevel = get(kwargs, :outputlevel, 0)
  sw = get(kwargs, :sweep, 1)

  maxdim = get(kwargs, :maxdim, typemax(Int))
  mindim = get(kwargs, :mindim, 1)
  cutoff = get(kwargs, :cutoff, 1E-16)
  noise = get(kwargs, :noise, 0.0)

  psi = copy(psi0)
  N = length(psi)

  if !isortho(psi) || orthocenter(psi) != 1
    orthogonalize!(psi, 1)
  end
  @assert isortho(psi) && orthocenter(psi) == 1

  position!(PH, psi, 1)

  maxtruncerr = 0.0

  for (b, ha) in sweepnext(N; ncenter=nsite)
    PH.nsite = nsite
    position!(PH, psi, b)

    if nsite == 1
      phi1 = psi[b]
    elseif nsite == 2
      phi1 = psi[b] * psi[b + 1]
    end

    phi1, info = solver(PH, time_step / 2, phi1)

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
      end

      PH.nsite = nsite - 1
      position!(PH, psi, b1)

      phi0, info = solver(PH, -time_step / 2, phi0)

      normalize && (phi0 ./= norm(phi0))

      if nsite == 2
        psi[b1] = phi0
      elseif nsite == 1
        psi[b + Δ] = phi0 * psi[b + Δ]
      end
      PH.nsite = nsite
    end

    if outputlevel >= 2
      @printf("Sweep %d, half %d, bond (%d,%d) \n", sw, ha, b, b + 1)
      @printf(
        "  Truncated using cutoff=%.1E maxdim=%d mindim=%d\n",
        cutoff,
        maxdim,
        mindim,
      )
      if spec != nothing
        @printf(
          "  Trunc. err=%.2E, bond dimension %d\n", spec.truncerr, dim(linkind(psi, b))
        )
      end
      flush(stdout)
    end

    sweep_is_done = (b == 1 && ha == 2)
    measure!(
      obs; psi, bond=b, sweep=sw, half_sweep=ha, spec, outputlevel, sweep_is_done
    )
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
  solver_kwargs = (;
    tol=get(kwargs, :solver_tol, 1E-12),
    maxiter=get(kwargs, :solver_krylovdim, 30),
    outputlevel=get(kwargs, :solver_outputlevel, 0),
  )
  function solver(H, t, psi0; kws...)
    psi, info = apply_exp(H, t, psi0; solver_kwargs..., kws...)
    return psi, info
  end
  return solver
end

function tdvp_solver(; kwargs...)
  solver_backend = get(kwargs,:solver_backend,"applyexp")
  if solver_backend=="applyexp"
    return applyexp_solver(; kwargs...)
  elseif solver_backend=="exponentiate"
    return exponentiate_solver(; kwargs...)
  else
    error("solver_backend=$solver_backend not recognized (options are \"applyexp\" or \"exponentiate\")")
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

function _tdvp_compute_sweeps(t; kwargs...)
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

  return Sweeps(
    nsweeps;
    maxdim=get(kwargs, :maxdim, typemax(Int)),
    mindim=get(kwargs, :mindim, 1),
    cutoff=get(kwargs, :cutoff, 1E-8),
    noise=get(kwargs, :noise, 0.0),
  )
end

function tdvp(solver, PH, t::Number, psi0::MPS, sweeps::Sweeps=Sweeps(); kwargs...)

  reverse_step = true
  isempty(sweeps) && (sweeps = _tdvp_compute_sweeps(t; kwargs...))
  time_step::Number = get(kwargs, :time_step, t)

  checkdone = get(kwargs, :checkdone, nothing)
  write_when_maxdim_exceeds::Union{Int,Nothing} = get(
    kwargs, :write_when_maxdim_exceeds, nothing
  )
  obs = get(kwargs, :observer, NoObserver())
  outputlevel::Int = get(kwargs, :outputlevel, 0)
  
  psi = copy(psi0)

  for sw in 1:nsweep(sweeps)

    if !isnothing(write_when_maxdim_exceeds) &&
      maxdim(sweeps, sw) > write_when_maxdim_exceeds
      if outputlevel >= 2
        println(
          "write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim(sweeps, sw) = $(maxdim(sweeps, sw)), writing environment tensors to disk",
        )
      end
      PH = disk(PH)
    end

    sw_time = @elapsed begin

      psi, PH, info = tdvp_iteration(solver, PH, time_step, psi; sweep=sw, reverse_step, kwargs...)

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

    if checkdone != nothing
      isdone = checkdone(; psi, sweep=sw, outputlevel, kwargs...)
    else
      isdone = checkdone!(obs; psi, sweep=sw, outputlevel)
    end
    isdone && break
  end
  return psi
end

function tdvp(H, t::Number, psi0::MPS, sweeps::Sweeps=Sweeps(); kwargs...)
  return tdvp(tdvp_solver(;kwargs...), H, t, psi0, sweeps; kwargs...)
end

function dmrg(H, psi0::MPS, sweeps::Sweeps=Sweeps(); kwargs...)
  t = Inf # DMRG is TDVP with an infinite timestep and no reverse step
  isempty(sweeps) && (sweeps = _tdvp_compute_sweeps(t; kwargs...))
  reverse_step = false
  psi = tdvp(eigsolve_solver(;kwargs...), H, t, psi0, sweeps; reverse_step, kwargs...)
  return psi
end

function dmrg(H::MPO, psi0::MPS, sweeps::Sweeps=Sweeps(); kwargs...)
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  H = ITensors.permute(H, (linkind, siteinds, linkind))
  PH = ProjMPO(H)
  return dmrg(PH, psi0, sweeps; kwargs...)
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
function tdvp(
  solver, Hs::Vector{MPO}, t::Number, psi0::MPS, sweeps::Sweeps=Sweeps(); kwargs...
)
  for H in Hs
    check_hascommoninds(siteinds, H, psi0)
    check_hascommoninds(siteinds, H, psi0')
  end
  Hs .= ITensors.permute.(Hs, Ref((linkind, siteinds, linkind)))
  PHs = ProjMPOSum(Hs)
  return tdvp(solver, PHs, t, psi0, sweeps; kwargs...)
end

function tdvp(H::Vector{MPO}, t::Number, psi0::MPS, sweeps::Sweeps=Sweeps(); kwargs...)
  return tdvp(tdvp_solver(;kwargs...), H, t, psi0, sweeps; kwargs...)
end

"""
    tdvp(H::MPO,psi0::MPS,t::Number; kwargs...)
    tdvp(H::MPO,psi0::MPS,t::Number,sweeps::Sweeps; kwargs...)

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
function tdvp(solver, H::MPO, t::Number, psi0::MPS, sweeps::Sweeps=Sweeps(); kwargs...)
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  H = ITensors.permute(H, (linkind, siteinds, linkind))
  PH = ProjMPO(H)
  return tdvp(solver, PH, t, psi0, sweeps; kwargs...)
end

function tdvp(H::MPO, t::Number, psi0::MPS, sweeps::Sweeps=Sweeps(); kwargs...)
  return tdvp(tdvp_solver(;kwargs...), H, t, psi0, sweeps; kwargs...)
end
