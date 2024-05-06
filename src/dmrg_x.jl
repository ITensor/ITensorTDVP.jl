using ITensors: MPS, array, contract, dag, uniqueind, onehot
using LinearAlgebra: eigen

function dmrg_x_solver(PH, t, psi0; current_time, outputlevel)
  H = contract(PH, ITensor(true))
  D, U = eigen(H; ishermitian=true)
  u = uniqueind(U, H)
  u′ = uniqueind(D, U)
  max_overlap, max_ind = findmax(abs, array(psi0 * dag(U)))
  U_max = U * dag(onehot(eltype(U), u => max_ind))
  D_max = D[u′ => max_ind, u => max_ind]
  return U_max, (; eigval=D_max)
end

function dmrg_x(
  PH, psi0::MPS; reverse_step=false, (observer!)=default_observer!(), kwargs...
)
  info_ref = Ref{Any}()
  info_observer! = values_observer(; info=info_ref)
  observer! = compose_observers(observer!, info_observer!)
  psi = alternating_update(dmrg_x_solver, PH, psi0; reverse_step, observer!, kwargs...)
  return info_ref[].eigval, psi
end
