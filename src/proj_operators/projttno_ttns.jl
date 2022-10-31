import ITensorNetworks:
  AbstractProjTTNO, make_environment!, _separate_first_two, environment
using Dictionaries: Dictionary, unset!, set!

mutable struct ProjTTNO_TTNS <: AbstractProjTTNO # questionable inheritance?
  PH::ProjTTNO
  Ms::Vector{ProjTTNS}
end

function ProjTTNO_TTNS(H::TTNO, M::TTNS)
  return ProjTTNO_TTNS(ProjTTNO(H), [ProjTTNS(M)])
end

function ProjTTNO_TTNS(H::MPO, Mv::Vector{MPS})
  return ProjTTNO_TTNS(ProjTTNO(H), [ProjTTNS(m) for m in Mv])
end

copy(P::ProjTTNO_TTNS) = ProjTTNO_TTNS(copy(P.PH), copy(P.Ms))

nsite(P::ProjTTNO_TTNS) = nsite(P.PH)

function set_nsite!(P::ProjTTNO_TTNS, nsite)
  set_nsite!(P.PH, nsite)
  for m in P.Ms
    set_nsite!(m, nsite)
  end
  return P
end

function position!(
  P::ProjTTNO_TTNS, psi::TTNS, pos::Union{Vector{<:Tuple},NamedDimEdge{Tuple}}
)
  position!(P.PH, psi, pos)
  for m in P.Ms
    position!(m, psi, pos)
  end
end

# function make_environment!(P:::ProjTTNO_TTNS, psi::TTNS, e::NamedDimEdge{Tuple})
#   make_environment!(P.PH, psi, e)
#   for m in P.Ms
#     make_environment!(m, psi, e)
#   end
#   return P
# end

contract(P::ProjTTNO_TTNS, v::ITensor) = contract(P.PH, v)

proj_ttns(P::ProjTTNO_TTNS) = [proj_ttns(m) for m in P.Ms]
