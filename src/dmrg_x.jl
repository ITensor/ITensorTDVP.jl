using ITensors: array, contract, dag, uniqueind, onehot
using ITensors.ITensorMPS: MPS
using LinearAlgebra: eigen

function eigen_updater(operator, state; current_time, time_step, outputlevel)
  contracted_operator = contract(operator, ITensor(true))
  d, u = eigen(contracted_operator; ishermitian=true)
  u_ind = uniqueind(u, contracted_operator)
  u′_ind = uniqueind(d, u)
  max_overlap, max_index = findmax(abs, array(state * dag(u)))
  u_max = u * dag(onehot(eltype(u), u_ind => max_index))
  d_max = d[u′_ind => max_index, u_ind => max_index]
  return u_max, (; eigval=d_max)
end

function dmrg_x(operator, state::MPS; updater=eigen_updater, (observer!)=default_observer(), kwargs...)
  info_ref = Ref{Any}()
  info_observer = values_observer(; info=info_ref)
  observer = compose_observers(observer!, info_observer)
  psi = alternating_update(
    operator, state; updater, (observer!)=observer, kwargs...
  )
  return info_ref[].eigval, psi
end
