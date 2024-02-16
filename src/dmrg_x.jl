using ITensors: MPS, array, contract, dag, uniqueind, onehot
using LinearAlgebra: eigen

function dmrg_x_solver(PH, t, psi0; current_time, outputlevel)
  H = contract(PH, ITensor(true))
  D, U = eigen(H; ishermitian=true)
  u = uniqueind(U, H)
  max_overlap, max_ind = findmax(abs, array(psi0 * dag(U)))
  U_max = U * dag(onehot(eltype(U), u => max_ind))
  return U_max, nothing
end

function dmrg_x(PH, psi0::MPS; reverse_step=false, kwargs...)
  t = ITensors.scalartype(psi0)(Inf)
  psi = tdvp(dmrg_x_solver, PH, t, psi0; reverse_step, kwargs...)
  return psi
end
