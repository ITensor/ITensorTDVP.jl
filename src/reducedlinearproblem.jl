using ITensors: contract
using ITensors.ITensorMPS: AbstractProjMPO, ProjMPO, makeL!, makeR!, nsite, set_nsite!

mutable struct ReducedLinearProblem <: AbstractProjMPO
  reduced_operator::ProjMPO
  reduced_constant_terms::Vector{ReducedConstantTerm}
end

function ReducedLinearProblem(operator::MPO, constant_term::MPS)
  return ReducedLinearProblem(ProjMPO(operator), [ReducedConstantTerm(constant_term)])
end

function ReducedLinearProblem(operator::MPO, constant_terms::Vector{MPS})
  return ReducedLinearProblem(ProjMPO(operator), ReducedConstantTerm.(constant_terms))
end

function Base.copy(reduced_problem::ReducedLinearProblem)
  return ReducedLinearProblem(
    copy(reduced_problem.reduced_operator), copy(reduced_problem.reduced_constant_terms)
  )
end

function ITensorMPS.nsite(reduced_problem::ReducedLinearProblem)
  return nsite(reduced_problem.reduced_operator)
end

function ITensorMPS.set_nsite!(reduced_problem::ReducedLinearProblem, nsite)
  set_nsite!(reduced_problem.reduced_operator, nsite)
  for m in reduced_problem.reduced_constant_terms
    set_nsite!(m, nsite)
  end
  return reduced_problem
end

function ITensorMPS.makeL!(reduced_problem::ReducedLinearProblem, state::MPS, position::Int)
  makeL!(reduced_problem.reduced_operator, state, k)
  for reduced_constant_term in reduced_problem.reduced_constant_terms
    makeL!(reduced_constant_term, state, k)
  end
  return reduced_problem
end

function ITensorMPS.makeR!(reduced_problem::ReducedLinearProblem, state::MPS, position::Int)
  makeR!(reduced_problem.reduced_operator, state, k)
  for reduced_constant_term in reduced_problem.reduced_constant_terms
    makeR!(reduced_constant_term, state, k)
  end
  return reduced_problem
end

function ITensors.contract(reduced_problem::ReducedLinearProblem, v::ITensor)
  return contract(reduced_problem.reduced_operator, v)
end

function reduced_constant_terms(reduced_problem::ReducedLinearProblem)
  return map(reduced_problem.reduced_constant_terms) do reduced_constant_term
    return contract(reduced_constant_term)
  end
end
