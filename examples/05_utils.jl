using ITensors: MPS, maxlinkdim
using ITensorTDVP: TDVPOrder, process_sweeps, tdvp_solver, tdvp_step, process_sweeps
using Observers: observer, update!
using Printf: @printf

function tdvp_nonuniform_timesteps(
  solver,
  PH,
  psi::MPS;
  time_steps,
  reverse_step=true,
  time_start=0.0,
  order=2,
  (step_observer!)=observer(),
  maxdim=ITensorTDVP.default_maxdim(),
  mindim=ITensorTDVP.default_mindim(),
  cutoff=ITensorTDVP.default_cutoff(),
  noise=ITensorTDVP.default_noise(),
  outputlevel=ITensorTDVP.default_outputlevel(),
  kwargs...,
)
  nsweeps = length(time_steps)
  maxdim, mindim, cutoff, noise = process_sweeps(; nsweeps, maxdim, mindim, cutoff, noise)
  tdvp_order = TDVPOrder(order, Base.Forward)
  current_time = time_start
  for sw in 1:nsweeps
    sw_time = @elapsed begin
      psi, PH, info = tdvp_step(
        tdvp_order,
        solver,
        PH,
        time_steps[sw],
        psi;
        kwargs...,
        current_time,
        reverse_step,
        sweep=sw,
        maxdim=maxdim[sw],
        mindim=mindim[sw],
        cutoff=cutoff[sw],
        noise=noise[sw],
      )
    end
    current_time += time_steps[sw]

    update!(step_observer!; psi, sweep=sw, outputlevel, current_time)

    if outputlevel ≥ 1
      print("After sweep ", sw, ":")
      print(" maxlinkdim=", maxlinkdim(psi))
      @printf(" maxerr=%.2E", info.maxtruncerr)
      print(" current_time=", round(current_time; digits=3))
      print(" time=", round(sw_time; digits=3))
      println()
      flush(stdout)
    end
  end
  return psi
end

function tdvp_nonuniform_timesteps(
  H,
  psi::MPS;
  ishermitian=ITensorTDVP.default_ishermitian(),
  issymmetric=ITensorTDVP.default_issymmetric(),
  solver_tol=ITensorTDVP.default_solver_tol(exponentiate),
  solver_krylovdim=ITensorTDVP.default_solver_krylovdim(exponentiate),
  solver_maxiter=ITensorTDVP.default_solver_maxiter(exponentiate),
  solver_outputlevel=ITensorTDVP.default_solver_outputlevel(exponentiate),
  kwargs...,
)
  return tdvp_nonuniform_timesteps(
    tdvp_solver(
      exponentiate;
      ishermitian,
      issymmetric,
      solver_tol,
      solver_krylovdim,
      solver_maxiter,
      solver_outputlevel,
    ),
    H,
    psi;
    kwargs...,
  )
end
