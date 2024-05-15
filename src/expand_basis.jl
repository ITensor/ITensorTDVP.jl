using ITensors:
  ITensors,
  Algorithm,
  Index,
  ITensor,
  @Algorithm_str,
  δ,
  commonind,
  dag,
  denseblocks,
  directsum,
  hasqns,
  prime,
  scalartype,
  uniqueinds
using ITensors.ITensorMPS: MPO, MPS, apply, dim, linkind, maxlinkdim, orthogonalize
using LinearAlgebra: normalize, svd, tr

#
# Possible improvements
#  - allow a maxdim argument to be passed to `extend`
#    and through `basis_extend`
#  - current behavior is letting bond dimension get too
#    big when used in imaginary time evolution
#  - Use (1-tau*operator)|state> to generate "Krylov" vectors
#    instead of operator|state>. Needed?
#

function expand_basis(state, reference; alg, kwargs...)
  return expand_basis(Algorithm(alg), state, reference; kwargs...)
end

"""
Given an MPS state and a collection of MPS references,
returns an MPS which is equal to state
(has fidelity 1.0 with state) but whose MPS basis
is expanded to contain a portion of the basis of
the references that is orthogonal to the MPS basis of state.
"""
function expand_basis(
  ::Algorithm"orthogonalize",
  state::MPS,
  references::Vector{MPS};
  cutoff=10^2 * eps(real(scalartype(state))),
)
  n = length(state)
  state = orthogonalize(state, n)
  references = map(reference -> orthogonalize(reference, n), references)
  s = siteinds(state)
  for j in reverse(2:n)
    # SVD state[j] to compute basisⱼ
    linds = [s[j - 1]; linkinds(state, j - 1)]
    _, λⱼ, basisⱼ = svd(state[j], linds; righttags="bψ_$j,Link")
    rinds = uniqueinds(basisⱼ, λⱼ)
    # Make projectorⱼ
    idⱼ = prod(r -> denseblocks(δ(scalartype(state), r', dag(r))), rinds)
    projectorⱼ = idⱼ - prime(basisⱼ, rinds) * dag(basisⱼ)
    # Sum reference density matrices
    ρⱼ = sum(reference -> prime(reference[j], rinds) * dag(reference[j]), references)
    # TODO: Fix bug that `tr` isn't preserving the element type.
    ρⱼ /= scalartype(state)(tr(ρⱼ))
    # Apply projectorⱼ
    ρⱼ_projected = apply(apply(projectorⱼ, ρⱼ), projectorⱼ)
    expanded_basisⱼ = basisⱼ
    if norm(ρⱼ_projected) > 10^3 * eps(real(scalartype(state)))
      # Diagonalize projected density matrix ρⱼ_projected
      # to compute reference_basisⱼ, which spans part of right basis
      # of references which is orthogonal to right basis of state
      dⱼ, reference_basisⱼ = eigen(
        ρⱼ_projected; cutoff, ishermitian=true, righttags="bϕ_$j,Link"
      )
      state_indⱼ = only(commoninds(basisⱼ, λⱼ))
      reference_indⱼ = only(commoninds(reference_basisⱼ, dⱼ))
      expanded_basisⱼ, bx = directsum(
        basisⱼ => state_indⱼ, reference_basisⱼ => reference_indⱼ
      )
    end
    # Shift ortho center one site left using dag(expanded_basisⱼ)
    # and replace tensor at site j with expanded_basisⱼ
    state[j - 1] = state[j - 1] * (state[j] * dag(expanded_basisⱼ))
    state[j] = expanded_basisⱼ
    for reference in references
      reference[j - 1] = reference[j - 1] * (reference[j] * dag(expanded_basisⱼ))
      reference[j] = expanded_basisⱼ
    end
  end
  return state
end

function expand_basis(
  ::Algorithm"global_krylov",
  state::MPS,
  operator::MPO;
  krylovdim=2,
  cutoff=(√(eps(real(scalartype(state))))),
)
  maxdim = maxlinkdim(state) + 1
  references = Vector{MPS}(undef, krylovdim)
  for k in 1:krylovdim
    prev = k == 1 ? state : references[k - 1]
    references[k] = normalize(apply(operator, prev; maxdim))
  end
  return expand_basis(state, references; alg="orthogonalize", cutoff)
end
