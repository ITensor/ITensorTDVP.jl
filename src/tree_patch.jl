# make MPS behave like a tree without actually converting it

import ITensorNetworks:
  NamedDimEdge,
  AbstractProjTTNO,
  nv,
  sites,
  underlying_graph,
  edgetype,
  set_ortho_center!,
  ortho_center,
  vertices
import ITensors:
  AbstractProjMPO,
  orthocenter,
  orthogonalize!,
  position!,
  set_ortho_lims!,
  tags,
  uniqueinds,
  siteinds

const TreeLikeState = Union{MPS,TTNS}
const TreeLikeOperator = Union{MPO,TTNO}
const TreeLikeProjOperator = Union{AbstractProjMPO,AbstractProjTTNO}
const TreeLikeProjOperatorSum = Union{ProjMPOSum,ProjTTNOSum}

# number of sites in state
nv(psi::AbstractMPS) = length(psi)

# support of effective hamiltonian
sites(P::AbstractProjMPO) = collect(ITensors.site_range(P))

# MPS lives on chain graph
underlying_graph(P::AbstractMPS) = chain_lattice_graph(length(P))
underlying_graph(P::AbstractProjMPO) = chain_lattice_graph(length(P.H))
underlying_graph(P::ProjMPOSum) = underlying_graph(P.pm[1])
vertices(psi::AbstractMPS) = vertices(underlying_graph(psi))

# default edgetype for ITensorNetworks
edgetype(::MPS) = NamedDimEdge{Tuple}

# catch-all constructors for projected operators
proj_operator(O::MPO) = ProjMPO(O)
proj_operator(O::TTNO) = ProjTTNO(O)
proj_operator_sum(Os::Vector{MPO}) = ProjMPOSum(Os)
proj_operator_sum(Os::Vector{TTNO}) = ProjTTNOSum(Os)
proj_operator_apply(psi0::MPS, O::MPO) = ProjMPOApply(psi0, O)
proj_operator_apply(psi0::TTNS, O::TTNO) = ProjTTNOApply(psi0, O)

# ortho lims as range versus ortho center as list of graph vertices
ortho_center(psi::MPS) = Tuple.(ortho_lims(psi))

function set_ortho_center!(psi::MPS, oc::Vector{<:Tuple})
  return set_ortho_lims!(psi, only(first(oc)):only(last(oc)))
end

# setting position of effective hamiltonian on graph
function position!(P::AbstractProjMPO, psi::MPS, pos::Vector{<:Tuple})
  return position!(P, psi, minimum(only.(pos)))
end
function position!(P::AbstractProjMPO, psi::MPS, pos::NamedDimEdge{Tuple})
  return position!(P, psi, maximum(only.(Tuple(pos))))
end
function position!(P::ProjMPOSum, psi::MPS, pos::Vector{<:Tuple})
  return position!(P, psi, minimum(only.(pos)))
end
function position!(P::ProjMPOSum, psi::MPS, pos::NamedDimEdge{Tuple})
  return position!(P, psi, maximum(only.(Tuple(pos))))
end

# link tags associated to a given graph edge
tags(psi::MPS, edge::NamedDimEdge{Tuple}) = tags(linkind(psi, minimum(only.(Tuple(edge)))))

# unique indices associated to the source of a graph edge
uniqueinds(psi::MPS, e::NamedDimEdge{Tuple}) = uniqueinds(psi[src(e)], psi[dst(e)])

# make tuple chain graph indices behave like integers
orthogonalize!(psi::MPS, v::Tuple; kwargs...) = orthogonalize!(psi, only(v; kwargs...))

Base.getindex(psi::MPS, v::Tuple) = psi[only(v)]

function Base.setindex!(psi::MPS, phi::ITensor, v::Tuple; kwargs...)
  return setindex!(psi, phi, only(v); kwargs...)
end

# site and link indices
siteinds(psi::AbstractMPS, v::Tuple; kwargs...) = siteinds(psi, only(v); kwargs...)

# Observocalypse

const ObserverLike = Union{Observer,ITensors.AbstractObserver}

function obs_update!(observer::ObserverLike, psi::MPS, pos::Vector{<:Tuple}; kwargs...)
  bond = minimum(only.(pos))
  return update!(observer; psi, bond, kwargs...)
end

function obs_update!(observer::ObserverLike, psi::MPS, pos::NamedDimEdge{Tuple}; kwargs...)
  return error("This should never be called!") # debugging...
end

function obs_update!(observer::ObserverLike, psi::TTNS, pos; kwargs...)
  return update!(observer; psi, pos, kwargs...)
end
