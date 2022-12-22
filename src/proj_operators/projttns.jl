# TODO

import ITensorNetworks:
  AbstractProjTTNO, make_environment!, _separate_first_two, environment
using Dictionaries: Dictionary, unset!, set!

mutable struct ProjTTNS <: AbstractProjTTNO # questionable inheritance?
  pos::Union{Vector{<:Tuple},NamedEdge{Tuple}}
  M::TTNS
  environments::Dictionary{NamedEdge{Tuple},ITensor}
end

# patch due to inheriting from AbstractProjTTNO while instances have no TTNO field...
underlying_graph(P::AbstractProjTTNS) = underlying_graph(P.M)

function ProjTTNS(M::TTNS)
  return ProjTTNS(eltype(H)[], M, Dictionary{edgetype(H),ITensor}())
end

function copy(P::ProjTTNS)
  return ProjTTNS(P.pos, copy(P.M), copy(P.environments))
end

# trivial if we choose to specify position as above; only kept to allow using alongside
# ProjMPS
function set_nsite!(P::ProjTTNS, nsite)
  return P
end

function make_environment!(P::ProjTTNS, psi::TTNS, e::NamedEdge{Tuple})::ITensor
  # invalidate environment for opposite edge direction if necessary
  reverse(e) ∈ incident_edges(P) || unset!(P.environments, reverse(e))
  # do nothing if valid envivalid environment already present
  if haskey(P.environments, e)
    env = environment(P, e)
  else
    if is_leaf(underlying_graph(P), src(e))
      # leaves are easy
      env = psi[src(e)] * dag(P.M[src(e)])
    else
      # construct by contracting neighbors
      neighbor_envs = ITensor[]
      for n in setdiff(neighbors(underlying_graph(P), src(e)), [dst(e)])
        push!(neighbor_envs, make_environment!(P, psi, edgetype(P)(n, src(e))))
      end
      # manually heuristic for contraction order: two environments, site tensors, then
      # other environments
      frst, scnd, rst = _separate_first_two(neighbor_envs)
      itensor_map = vcat(psi[src(e)], frst, scnd, dag(P.M[src(e)]), rst)
      # TODO: actually use optimal contraction sequence here
      env = reduce(*, itensor_map)
    end
    # cache
    set!(P.environments, e, env)
  end
  @assert(
    hascommoninds(environment(P, e), psi[src(e)]),
    "Something went wrong, probably re-orthogonalized this edge in the same direction twice!"
  )
  return env
end

function contract(P::ProjTTNS, v::ITensor)::ITensor
  environments = ITensor[environment(P, edge) for edge in incident_edges(P)]
  if on_edge(P)
    itensor_map = environments
  else
    itensor_map = Union{ITensor,OneITensor}[] # TODO: will any of these ever be a OneITensor?
    for s in sites(P)
      site_envs = filter(hascommoninds(P.M[s]), environments)
      frst, scnd, rst = _separate_first_two(site_envs)
      site_tensors = vcat(frst, scnd, dag(P.M[s]), rst)
      append!(itensor_map, site_tensors)
    end
  end
  # TODO: actually use optimal contraction sequence here
  Mv = v
  for it in itensor_map
    Mv *= it
  end
  return Mv
end

function contract(P::ProjTTNS, v::ITensor)
  itensor_map = Union{ITensor,OneITensor}[lproj(P)]
  append!(itensor_map, [prime(t, "Link") for t in P.M[site_range(P)]])
  push!(itensor_map, rproj(P))

  # Reverse the contraction order of the map if
  # the first tensor is a scalar (for example we
  # are at the left edge of the system)
  if dim(first(itensor_map)) == 1
    reverse!(itensor_map)
  end

  # Apply the map
  Mv = v
  for it in itensor_map
    Mv *= it
  end
  return Mv
end

function proj_ttns(P::ProjTTNS)
  return contract(P, ITensor(1)) # ?
end
