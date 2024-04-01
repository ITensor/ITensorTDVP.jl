module ITensorTDVP

using ITensors: ITensors, MPO
using ITensors.NDTensors: @Algorithm_str
using ITensors.ITensorMPS: ITensorMPS
using KrylovKit: KrylovKit

tdvp(args...; kwargs...) = ITensorMPS.itensortdvp_tdvp(args...; kwargs...)

function KrylovKit.linsolve(A::MPO, args...; kwargs...)
  return ITensorMPS.itensortdvp_linsolve(A, args...; kwargs...)
end

dmrg(args...; kwargs...) = ITensorMPS.itensortdvp_dmrg(args...; kwargs...)

dmrg_x(args...; kwargs...) = ITensorMPS.itensortdvp_dmrg_x(args...; kwargs...)

function ITensors.contract(alg::Algorithm"fit", args...; kwargs...)
  return ITensorMPS.itensortdvp_contract(alg, args...; kwargs...)
end

include("solver_utils.jl")
export TimeDependentSum, to_vec

end
