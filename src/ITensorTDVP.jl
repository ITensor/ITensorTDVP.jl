module ITensorTDVP

export TimeDependentSum, dmrg_x, linsolve, tdvp, to_vec

using ITensors: ITensors
using ITensors.ITensorMPS: ITensorMPS, MPO, MPS
using ITensors.NDTensors: @Algorithm_str
using KrylovKit: KrylovKit

function tdvp(operator, t::Number, init_state::MPS; kwargs...)
  return ITensorMPS.itensortdvp_tdvp(operator, t, init_state; kwargs...)
end
function tdvp(t::Number, operator, init_state::MPS; kwargs...)
  return ITensorMPS.itensortdvp_tdvp(t, operator, init_state; kwargs...)
end
function tdvp(operator, init_state::MPS, t::Number; kwargs...)
  return ITensorMPS.itensortdvp_tdvp(operator, init_state, t; kwargs...)
end
function tdvp(solver, operator::MPO, t::Number, init_state::MPS; kwargs...)
  return ITensorMPS.itensortdvp_tdvp(solver, operator, t, init_state; kwargs...)
end
function tdvp(solver, t::Number, operator::MPO, init_state::MPS; kwargs...)
  return ITensorMPS.itensortdvp_tdvp(solver, t, operator, init_state; kwargs...)
end
function tdvp(solver, init_state::MPS, t::Number, operator::MPO; kwargs...)
  return ITensorMPS.itensortdvp_tdvp(solver, init_state, t, operator; kwargs...)
end
function tdvp(solver, operators::Vector{MPO}, t::Number, init_state::MPS; kwargs...)
  return ITensorMPS.itensortdvp_tdvp(solver, operators, t, init_state; kwargs...)
end

function KrylovKit.linsolve(
  A::MPO, b::MPS, x0::MPS, a0::Number=false, a1::Number=true; kwargs...
)
  return ITensorMPS.itensortdvp_linsolve(A, b, x0, a0, a1; kwargs...)
end

using ITensors.ITensorMPS: ITensorMPS, MPS
function dmrg(operator, init_state::MPS; kwargs...)
  return ITensorMPS.itensortdvp_dmrg(operator, init_state; kwargs...)
end

function dmrg_x(operator, init_state::MPS; kwargs...)
  return ITensorMPS.itensortdvp_dmrg_x(operator, init_state; kwargs...)
end

function ITensors.contract(alg::Algorithm"fit", tn1::MPO, tn2::MPS; kwargs...)
  return ITensorMPS.itensortdvp_contract(alg, tn1, tn2; kwargs...)
end

include("solver_utils.jl")

end
