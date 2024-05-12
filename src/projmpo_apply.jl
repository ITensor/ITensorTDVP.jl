using ITensors: ITensor
using ITensors.ITensorMPS: ITensorMPS, AbstractProjMPO, MPO, MPS

"""
A ProjMPOApply represents the application of an
MPO `operator` onto an MPS `input_state` but "projected" by
the basis of a different MPS `state` (which
could be an approximation to operator|state>).

As an implementation of the AbstractProjMPO
type, it supports multiple `nsite` values for
one- and two-site algorithms.

```
     *--*--*-      -*--*--*--*--*--* <state|
     |  |  |  |  |  |  |  |  |  |  |
     h--h--h--h--h--h--h--h--h--h--h operator  
     |  |  |  |  |  |  |  |  |  |  |
     o--o--o-      -o--o--o--o--o--o |input_state>
```
"""
mutable struct ProjMPOApply <: AbstractProjMPO
  lpos::Int
  rpos::Int
  nsite::Int
  input_state::MPS
  operator::MPO
  LR::Vector{ITensor}
end

function ProjMPOApply(input_state::MPS, operator::MPO)
  lpos = 0
  rpos = length(operator) + 1
  nsite = 2
  LR = Vector{ITensor}(undef, length(operator))
  return ProjMPOApply(
    lpos,
    rpos,
    nsite,
    input_state,
    operator,
    LR,
  )
end

function Base.getproperty(reduced_operator::ProjMPOApply, sym::Symbol)
  if sym === :H
    # This is for compatibility with `AbstractProjMPO`.
    # TODO: Don't use `reduced_operator.H` in `AbstractProjMPO`.
    return getfield(reduced_operator, :operator)
  end
  return getfield(reduced_operator, sym)
end

function Base.copy(reduced_operator::ProjMPOApply)
  return ProjMPOApply(
    reduced_operator.lpos,
    reduced_operator.rpos,
    reduced_operator.nsite,
    copy(reduced_operator.input_state),
    copy(reduced_operator.operator),
    copy(reduced_operator.LR),
  )
end

Base.length(reduced_operator::ProjMPOApply) = length(reduced_operator.operator)

function ITensorMPS.set_nsite!(reduced_operator::ProjMPOApply, nsite)
  reduced_operator.nsite = nsite
  return reduced_operator
end

function ITensorMPS.makeL!(reduced_operator::ProjMPOApply, state::MPS, k::Int)
  # Save the last `L` that is made to help with caching
  # for DiskProjMPO
  ll = reduced_operator.lpos
  if ll ≥ k
    # Special case when nothing has to be done.
    # Still need to change the position if lproj is
    # being moved backward.
    reduced_operator.lpos = k
    return nothing
  end
  # Make sure ll is at least 0 for the generic logic below
  ll = max(ll, 0)
  L = lproj(reduced_operator)
  while ll < k
    L =
      L *
      reduced_operator.input_state[ll + 1] *
      reduced_operator.operator[ll + 1] *
      dag(state[ll + 1])
    reduced_operator.LR[ll + 1] = L
    ll += 1
  end
  # Needed when moving lproj backward.
  reduced_operator.lpos = k
  return reduced_operator
end

function ITensorMPS.makeR!(reduced_operator::ProjMPOApply, state::MPS, k::Int)
  # Save the last `R` that is made to help with caching
  # for DiskProjMPO
  rl = reduced_operator.rpos
  if rl ≤ k
    # Special case when nothing has to be done.
    # Still need to change the position if rproj is
    # being moved backward.
    reduced_operator.rpos = k
    return nothing
  end
  N = length(reduced_operator.operator)
  # Make sure rl is no bigger than `N + 1` for the generic logic below
  rl = min(rl, N + 1)
  R = rproj(reduced_operator)
  while rl > k
    R =
      R *
      reduced_operator.input_state[rl - 1] *
      reduced_operator.operator[rl - 1] *
      dag(state[rl - 1])
    reduced_operator.LR[rl - 1] = R
    rl -= 1
  end
  reduced_operator.rpos = k
  return reduced_operator
end
