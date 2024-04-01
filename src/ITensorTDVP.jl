module ITensorTDVP

using ITensors: ITensors, MPO, MPS
using ITensors.NDTensors: @Algorithm_str
using ITensors.ITensorMPS: ITensorMPS
using KrylovKit: KrylovKit

function tdvp(H_generic, t::Number, psi0::MPS; kwargs...)
  return ITensorMPS.itensortdvp_tdvp(H_generic, t, psi0; kwargs...)
end
function tdvp(t::Number, H_generic, psi0::MPS; kwargs...)
  return ITensorMPS.itensortdvp_tdvp(t, H_generic, psi0; kwargs...)
end
function tdvp(H_generic, psi0::MPS, t::Number; kwargs...)
  return ITensorMPS.itensortdvp_tdvp(H_generic, psi0, t; kwargs...)
end
function tdvp(solver, H::MPO, t::Number, psi0::MPS; kwargs...)
  return ITensorMPS.itensortdvp_tdvp(solver, H, t, psi0; kwargs...)
end
function tdvp(solver, t::Number, H::MPO, psi0::MPS; kwargs...)
  return ITensorMPS.itensortdvp_tdvp(solver, t, H, psi0; kwargs...)
end
function tdvp(solver, psi0::MPS, t::Number, H::MPO; kwargs...)
  return ITensorMPS.itensortdvp_tdvp(solver, psi0, t, H; kwargs...)
end
function tdvp(solver, Hs::Vector{MPO}, t::Number, psi0::MPS; kwargs...)
  return ITensorMPS.itensortdvp_tdvp(solver, Hs, t, psi0; kwargs...)
end

function KrylovKit.linsolve(
  A::MPO, b::MPS, x0::MPS, a0::Number=false, a1::Number=true; kwargs...
)
  return ITensorMPS.itensortdvp_linsolve(A, b, x0, a0, a1; kwargs...)
end

dmrg(H, psi0::MPS; kwargs...) = ITensorMPS.itensortdvp_dmrg(H, psi0; kwargs...)

dmrg_x(H, psi0::MPS; kwargs...) = ITensorMPS.itensortdvp_dmrg_x(H, psi0; kwargs...)

function ITensors.contract(alg::Algorithm"fit", A::MPO, psi::MPS; kwargs...)
  return ITensorMPS.itensortdvp_contract(alg, A, psi; kwargs...)
end

include("solver_utils.jl")
export TimeDependentSum, to_vec

end
