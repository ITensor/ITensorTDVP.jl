module ITensorTDVP

using ITensors: ITensors
using ITensors.ITensorMPS: ITensorMPS, MPO, MPS
using ITensors.NDTensors: @Algorithm_str
using KrylovKit: KrylovKit

function tdvp(operator, t::Number, initial_state::MPS; kwargs...)
  return ITensorMPS.itensortdvp_tdvp(operator, t, initial_state; kwargs...)
end
function tdvp(t::Number, operator, initial_state::MPS; kwargs...)
  return ITensorMPS.itensortdvp_tdvp(t, operator, initial_state; kwargs...)
end
function tdvp(operator, initial_state::MPS, t::Number; kwargs...)
  return ITensorMPS.itensortdvp_tdvp(operator, initial_state, t; kwargs...)
end
function tdvp(solver, operator::MPO, t::Number, initial_state::MPS; kwargs...)
  return ITensorMPS.itensortdvp_tdvp(solver, operator, t, initial_state; kwargs...)
end
function tdvp(solver, t::Number, operator::MPO, initial_state::MPS; kwargs...)
  return ITensorMPS.itensortdvp_tdvp(solver, t, operator, initial_state; kwargs...)
end
function tdvp(solver, initial_state::MPS, t::Number, operator::MPO; kwargs...)
  return ITensorMPS.itensortdvp_tdvp(solver, initial_state, t, operator; kwargs...)
end
function tdvp(solver, operators::Vector{MPO}, t::Number, initial_state::MPS; kwargs...)
  return ITensorMPS.itensortdvp_tdvp(solver, operators, t, initial_state; kwargs...)
end

function KrylovKit.linsolve(
  A::MPO, b::MPS, x0::MPS, a0::Number=false, a1::Number=true; kwargs...
)
  return ITensorMPS.itensortdvp_linsolve(A, b, x0, a0, a1; kwargs...)
end

function dmrg(operator, initial_state::MPS; kwargs...)
  return ITensorMPS.itensortdvp_dmrg(operator, initial_state; kwargs...)
end

function dmrg_x(operator, initial_state::MPS; kwargs...)
  return ITensorMPS.itensortdvp_dmrg_x(operator, initial_state; kwargs...)
end

function ITensors.contract(alg::Algorithm"fit", A::MPO, psi::MPS; kwargs...)
  return ITensorMPS.itensortdvp_contract(alg, A, psi; kwargs...)
end

include("solver_utils.jl")
export TimeDependentSum, to_vec

end
