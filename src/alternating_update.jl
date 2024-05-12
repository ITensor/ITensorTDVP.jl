using ITensors: ITensors, permute
using ITensors.ITensorMPS:
  ## AbstractObserver,
  MPO,
  MPS,
  ProjMPO,
  ProjMPOSum,
  check_hascommoninds,
  ## checkdone!,
  disk,
  linkind,
  maxlinkdim,
  siteinds

## function _compute_nsweeps(t; time_step=default_time_step(t), nsweeps=default_nsweeps())
## 
##   @show t, time_step, nsweeps
## 
##   if isinf(t) && isnothing(nsweeps)
##     nsweeps = 1
##   elseif !isnothing(nsweeps) && time_step != t
##     error("Cannot specify both time_step and nsweeps in alternating_update")
##   elseif isfinite(time_step) && abs(time_step) > 0 && isnothing(nsweeps)
##     nsweeps = convert(Int, ceil(abs(t / time_step)))
##     if !(nsweeps * time_step ≈ t)
##       error("Time step $time_step not commensurate with total time t=$t")
##     end
##   end
## 
##   @show nsweeps
## 
##   return nsweeps
## end

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

function process_sweeps(; nsweeps, maxdim, mindim, cutoff, noise)
  maxdim = _extend_sweeps_param(maxdim, nsweeps)
  mindim = _extend_sweeps_param(mindim, nsweeps)
  cutoff = _extend_sweeps_param(cutoff, nsweeps)
  noise = _extend_sweeps_param(noise, nsweeps)
  return (; maxdim, mindim, cutoff, noise)
end

function alternating_update(
  solver,
  reduced_operator,
  init::MPS;
  nsweeps=default_nsweeps(),
  checkdone=default_checkdone(),
  write_when_maxdim_exceeds=default_write_when_maxdim_exceeds(),
  nsite=default_nsite(),
  reverse_step=default_reverse_step(),
  time_start=default_time_start(),
  time_step=default_time_step(),
  order=default_order(),
  (observer!)=default_observer!(),
  (step_observer!)=default_step_observer!(),
  outputlevel=default_outputlevel(),
  normalize=default_normalize(),
  maxdim=default_maxdim(),
  mindim=default_mindim(),
  cutoff=default_cutoff(ITensors.scalartype(init)),
  noise=default_noise(),
)
  ## nsweeps = _compute_nsweeps(t; time_step, nsweeps)
  maxdim, mindim, cutoff, noise = process_sweeps(; nsweeps, maxdim, mindim, cutoff, noise)
  forward_order = TDVPOrder(order, Base.Forward)
  state = copy(init)
  # Keep track of the start of the current time step.
  # Helpful for tracking the total time, for example
  # when using time-dependent solvers.
  # This will be passed as a keyword argument to the
  # `solver`.
  current_time = time_start
  info = nothing
  for sweep in 1:nsweeps
    if !isnothing(write_when_maxdim_exceeds) && maxdim[sweep] > write_when_maxdim_exceeds
      if outputlevel >= 2
        println(
          "write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim(sweeps, sw) = $(maxdim(sweeps, sweep)), writing environment tensors to disk",
        )
      end
      reduced_operator = disk(reduced_operator)
    end
    sweep_elapsed_time = @elapsed begin
      state, reduced_operator, info = sweep_update(
        forward_order,
        solver,
        reduced_operator,
        state;
        nsite,
        current_time,
        time_step,
        reverse_step,
        sweep,
        observer!,
        normalize,
        maxdim=maxdim[sweep],
        mindim=mindim[sweep],
        cutoff=cutoff[sweep],
        noise=noise[sweep],
      )
    end
    if !isnothing(time_step)
      current_time += time_step
    end
    update_observer!(step_observer!; state, sweep, outputlevel, current_time)
    if outputlevel >= 1
      print("After sweep ", sweep, ":")
      print(" maxlinkdim=", maxlinkdim(state))
      @printf(" maxerr=%.2E", info.maxtruncerr)
      print(" current_time=", round(current_time; digits=3))
      print(" time=", round(sweep_elapsed_time; digits=3))
      println()
      flush(stdout)
    end
    isdone = checkdone(; state, sweep, outputlevel)
    ## isdone = false
    ## if !isnothing(checkdone)
    ##  isdone = checkdone(; state, sweep, outputlevel)
    ## elseif observer! isa AbstractObserver
    ##   isdone = checkdone!(observer!; state, sweep, outputlevel)
    ## end
    isdone && break
  end
  return state
end

# Convenience wrapper to not have to specify time step.
# Use a time step of `Inf` as a convention, since TDVP
# with an infinite time step corresponds to DMRG.
## function alternating_update(solver, operator, init::MPS; kwargs...)
##   return alternating_update(solver, operator, ITensors.scalartype(init)(Inf), init; kwargs...)
## end

function alternating_update(solver, operator::MPO, init::MPS; kwargs...)
  check_hascommoninds(siteinds, operator, init)
  check_hascommoninds(siteinds, operator, init')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  operator = permute(operator, (linkind, siteinds, linkind))
  reduced_operator = ProjMPO(operator)
  return alternating_update(solver, reduced_operator, init; kwargs...)
end

function alternating_update(solver, operators::Vector{MPO}, init::MPS; kwargs...)
  for operator in operators
    check_hascommoninds(siteinds, operator, init)
    check_hascommoninds(siteinds, operator, init')
  end
  operators .= ITensors.permute.(operators, Ref((linkind, siteinds, linkind)))
  reduced_operator = ProjMPOSum(operators)
  return alternating_update(solver, reduced_operator, init; kwargs...)
end
