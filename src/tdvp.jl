
"""
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
function tdvp(H::MPO, psi0::MPS, t::Number, sweeps::Sweeps; kwargs...)::MPS
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  H = ITensors.permute(H, (linkind, siteinds, linkind))
  PH = ProjMPO(H)
  return tdvp(PH, psi0, t, sweeps; kwargs...)
end

"""
    tdvp(Hs::Vector{MPO},psi0::MPS,t::Number,sweeps::Sweeps;kwargs...)

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
function tdvp(Hs::Vector{MPO}, psi0::MPS, t::Number, sweeps::Sweeps; kwargs...)::MPS
  for H in Hs
    check_hascommoninds(siteinds, H, psi0)
    check_hascommoninds(siteinds, H, psi0')
  end
  Hs .= ITensors.permute.(Hs, Ref((linkind, siteinds, linkind)))
  PHS = ProjMPOSum(Hs)
  return tdvp(PHS, psi0, t, sweeps; kwargs...)
end


function tdvp(PH,
              psi0::MPS,
              t::Number,
              sweeps;
              kwargs...)::MPS

  if length(psi0) == 1
    error(
      "`tdvp` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.exponentiate`, etc.",
    )
  end

  @debug_check begin
    # Debug level checks
    # Enable with ITensors.enable_debug_checks()
    checkflux(psi0)
    checkflux(PH)
  end

  nsite::Int = get(kwargs, :nsite, 2)
  normalize::Bool = get(kwargs, :normalize, false)
  outputlevel::Int = get(kwargs, :outputlevel, 0)
  which_decomp::Union{String,Nothing} = get(kwargs, :which_decomp, nothing)
  svd_alg::String = get(kwargs, :svd_alg, "divide_and_conquer")
  obs = get(kwargs, :observer, NoObserver())

  write_when_maxdim_exceeds::Union{Int,Nothing} = get(
    kwargs, :write_when_maxdim_exceeds, nothing
  )

  # exponentiate kwargs
  exponentiate_tol::Float64 = get(kwargs, :exponentiate_tol, 1e-14)
  exponentiate_krylovdim::Int = get(kwargs, :exponentiate_krylovdim, 20)
  exponentiate_maxiter::Int = get(kwargs, :exponentiate_maxiter, 1)
  exponentiate_verbosity::Int = get(kwargs, :exponentiate_verbosity, 0)

  psi = copy(psi0)
  N = length(psi)

  if !isortho(psi) || orthocenter(psi) != 1
    orthogonalize!(psi, 1)
  end
  @assert isortho(psi) && orthocenter(psi) == 1

  position!(PH, psi, 1)

  for sw in 1:nsweep(sweeps)
    maxtruncerr = 0.0

    if !isnothing(write_when_maxdim_exceeds) &&
      maxdim(sweeps, sw) > write_when_maxdim_exceeds
      if outputlevel >= 2
        println(
          "write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim(sweeps, sw) = $(maxdim(sweeps, sw)), writing environment tensors to disk",
        )
      end
      PH = disk(PH)
    end

    for (b, ha) in sweepnext(N; ncenter=nsite)
      PH.nsite = nsite
      position!(PH, psi, b)

      if nsite == 1
        phi1 = psi[b]
      elseif nsite == 2
        phi1 = psi[b]*psi[b+1]
      end

      phi1, info = exponentiate(PH, t/2, phi1; 
                               tol=exponentiate_tol,
                               krylovdim=exponentiate_krylovdim,
                               maxiter=exponentiate_maxiter)

      do_normalize && (phi1 /= norm(phi1))

      spec = nothing
      if nsite == 1
        psi[b] = phi1
      elseif nsite == 2
        ortho = ha == 1 ? "left" : "right"

        drho = nothing
        if noise(sweeps, sw) > 0.0 && ha==1
          drho = noise(sweeps, sw) * noiseterm(PH, phi, ortho)
        end

        spec = replacebond!(
          psi,
          b,
          phi1;
          maxdim=maxdim(sweeps, sw),
          mindim=mindim(sweeps, sw),
          cutoff=cutoff(sweeps, sw),
          eigen_perturbation=drho,
          ortho=ortho,
          normalize=do_normalize,
          which_decomp=which_decomp,
          svd_alg=svd_alg,
        )
        maxtruncerr = max(maxtruncerr, spec.truncerr)
      end


      #
      # Do backwards evolution step
      #
      if (ha==1 && b!=N) || (ha==2 && b!=1)
        b1 = (ha == 1 ? b+1 : b)
        Δ = (ha==1 ? +1 : -1)
        if nsite == 2
          phi0 = psi[b1]
        elseif nsite == 1
          uinds = uniqueinds(phi1,psi[b+Δ])
          U,S,V = svd(phi1,uinds)
          psi[b] = U
          phi0 = S*V
        end

        PH.nsite = nsite-1
        position!(PH,psi,b1)
        phi0,info = exponentiate(PH, -t/2, phi0; 
                                 tol=exponentiate_tol,
                                 krylovdim=exponentiate_krylovdim,
                                 maxiter=exponentiate_maxiter)
        do_normalize && (phi0 ./= norm(phi0))

        if nsite == 2
          psi[b1] = phi0
        elseif nsite == 1
          psi[b+Δ] = phi0*psi[b+Δ]
        end
        PH.nsite = nsite
      end

      if outputlevel >= 2
        @printf(
          "Sweep %d, half %d, bond (%d,%d) \n", sw, ha, b, b + 1
        )
        @printf(
          "  Truncated using cutoff=%.1E maxdim=%d mindim=%d\n",
          cutoff(sweeps, sw),
          maxdim(sweeps, sw),
          mindim(sweeps, sw)
        )
        @printf(
          "  Trunc. err=%.2E, bond dimension %d\n", spec.truncerr, dim(linkind(psi, b))
        )
        flush(stdout)
      end

      sweep_is_done = (b == 1 && ha == 2)
      measure!(
        obs;
        psi=psi,
        bond=b,
        sweep=sw,
        half_sweep=ha,
        spec=spec,
        outputlevel=outputlevel,
        sweep_is_done=sweep_is_done,
      )

    end

    if outputlevel >= 1
      @printf(
        "After sweep %d maxlinkdim=%d maxerr=%.2E time=%.3f\n",
        sw,
        maxlinkdim(psi),
        maxtruncerr,
        sw_time
      )
      flush(stdout)
    end
    isdone = checkdone!(obs; psi=psi, sweep=sw, outputlevel=outputlevel)

    isdone && break
  end

  return psi
end


function _tdvp_sweeps(; nsweeps=1, maxdim=typemax(Int), mindim=1, cutoff=1E-8, noise=0.0, kwargs...)
  sweeps = Sweeps(nsweeps)
  setmaxdim!(sweeps, maxdim...)
  setmindim!(sweeps, mindim...)
  setcutoff!(sweeps, cutoff...)
  setnoise!(sweeps, noise...)
  return sweeps
end

function tdvp(x1, x2, psi0::MPS, t::Number; kwargs...)
  return tdvp(x1, x2, psi0, t, _tdvp_sweeps(; kwargs...); kwargs...)
end

function tdvp(x1, psi0::MPS, t::Number; kwargs...)
  return tdvp(x1, psi0, t, _tdvp_sweeps(; kwargs...); kwargs...)
end
