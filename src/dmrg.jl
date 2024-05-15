using ITensors: ITensors
using ITensors.ITensorMPS: MPS
using KrylovKit: eigsolve

function eigsolve_updater(
  operator,
  init;
  internal_kwargs,
  which_eigval=:SR,
  ishermitian=true,
  tol=10^2 * eps(real(ITensors.scalartype(init))),
  krylovdim=3,
  maxiter=1,
  verbosity=0,
  eager=false,
)
  howmany = 1
  eigvals, eigvecs, info = eigsolve(
    operator, init, howmany, which_eigval; ishermitian, tol, krylovdim, maxiter, verbosity
  )
  return eigvecs[1], (; info, eigval=eigvals[1])
end

function dmrg(
  operator, init::MPS; updater=eigsolve_updater, (observer!)=default_observer(), kwargs...
)
  info_ref! = Ref{Any}()
  info_observer! = values_observer(; info=info_ref!)
  observer! = compose_observers(observer!, info_observer!)
  state = alternating_update(operator, init; updater, observer!, kwargs...)
  return info_ref![].eigval, state
end
