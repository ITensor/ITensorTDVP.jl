function replaceind_indval(IV::Tuple, iĩ::Pair)
  i, ĩ = iĩ
  return ntuple(n -> first(IV[n]) == i ? ĩ => last(IV[n]) : IV[n], length(IV))
end

# atol controls the tolerance cutoff for determining which eigenvectors are in the null
# space of the isometric MPS tensors. Setting to 1e-2 since we only want to keep
# the eigenvectors corresponding to eigenvalues of approximately 1.

###the most general implementation may be passing MPS, lims, and bondtensor (defaulting to the Id-matrix)
###then we can tensorsum both left and right tensor, and just apply the bondtensor to restore input gauge
###ideally we would also be able to apply tensorsum to the bond tensor instead of that loop below
"""
expand subspace (2-site like) in a sweep 
"""
function subspace_expansion_sweep!(
  ψ::MPS, PH::Union{ProjMPO,ProjMPOSum}; maxdim, cutoff, atol=1e-10, kwargs...
)
  N = length(ψ)

  if !isortho(ψ) || orthocenter(ψ) != 1
    orthogonalize!(ψ, 1)
  end

  PH.nsite = 2
  nsite = 2
  position!(PH, ψ, 1)

  for (b, ha) in sweepnext(N; ncenter=2)

    ##TODO: figure out whether these calls should be here or inside subspace expansion, currently we do both?
    orthogonalize!(ψ, b)
    position!(PH, ψ, b)

    if (ha == 1 && (b + nsite - 1 != N)) || (ha == 2 && b != 1)
      b1 = (ha == 1 ? b + 1 : b)
      Δ = (ha == 1 ? +1 : -1)
      inds = (ha == 1 ? (b, b + Δ) : (b + Δ, b))
      subspace_expansion!(
        ψ, PH, (ψ.llim, ψ.rlim), inds; maxdim, cutoff=cutoff, atol=atol, kwargs...
      )
    end
  end
  return nothing
end

function subspace_expansion!(
  ψ::MPS,
  PH,
  lims::Tuple{Int,Int},
  b::Tuple{Int,Int};
  bondtensor=nothing,
  maxdim,
  cutoff,
  atol=1e-10,
  kwargs...,
)
  """
  expands in nullspace of site-tensors b
  """
  #atol refers to the tolerance in nullspace determination (for finite MPS can probably be set rather small)
  #cutoff refers to largest singular value of gradient (acceleration of population gain of expansion vectors) to keep
  #this should be related to the cutoff in two-site TDVP: \Delta_rho_i = 0.5 * lambda_y * tau **2 
  #note that in the initial SVD there is another cutoff parameter, that we set to roughly machine precision for the time being
  #(allows to move bond-dimension between different partial QN sectors, if no such truncation occurs distribution of bond-dimensions
  #between different QNs locally is static once bond dimension saturates maxdim.)

  ##this should only work for the case where rlim-llim > 1
  ##not a valid MPS otherwise anyway (since bond matrix not part of MPS unless so defined like in VidalMPS struct)
  llim, rlim = lims
  n1, n2 = b
  PH.nsite = 2
  old_linkdim = dim(commonind(ψ[n1], ψ[n2]))
  cutoff_compress = get(kwargs, :cutoff_compress, 1e-12)

  ###move orthogonality center to bond, check whether there are vanishing contributions to the wavefunctions and truncate accordingly
  if llim == n1
    @assert rlim == n2 + 1
    U, S, V = svd(
      ψ[n2], uniqueinds(ψ[n2], ψ[n1]); maxdim=maxdim, cutoff=cutoff_compress, kwargs...
    )  ##lookup svd interface again
    ϕ_1 = ψ[n1] * V
    ϕ_2 = U
    old_linkdim = dim(commonind(U, S))
    bondtensor = S

  elseif rlim == n2
    @assert llim == n1 - 1
    U, S, V = svd(
      ψ[n1], uniqueinds(ψ[n1], ψ[n2]); maxdim=maxdim, cutoff=cutoff_compress, kwargs...
    )
    ϕ_1 = U
    ϕ_2 = ψ[n2] * V
    old_linkdim = dim(commonind(U, S))
    bondtensor = S
  end

  ###don't expand if we are already at maxdim
  if old_linkdim >= maxdim
    println("not expanding")
    return nothing
  end
  position!(PH, ψ, min(n1, n2))

  #orthogonalize(ψ,n1)
  linkind_l = commonind(ϕ_1, bondtensor)
  linkind_r = commonind(ϕ_2, bondtensor)

  NL = nullspace(ϕ_1, linkind_l; atol=atol)
  NR = nullspace(ϕ_2, linkind_r; atol=atol)

  ###NOTE: This will fail for rank-1 trees with physical DOFs on leafs
  ###NOTE: one-sided subspace expansion seems to not work well at least for trees according to Lachlan Lindoy
  if norm(NL) == 0.0 || norm(NR) == 0.0
    return bondtensor
  end

  ###form 2site wavefunction
  ϕ = ϕ_1 * bondtensor * ϕ_2

  ###get subspace expansion
  newL, S, newR, success = _subspace_expand_core(
    ϕ, PH, NL, NR, ; maxdim=maxdim - old_linkdim, cutoff, kwargs...
  )
  @assert success
  #@show expanS
  if success == false
    return nothing
  end

  ###add expansion direction to current site tensors
  ALⁿ¹, newl = ITensors.directsum(
    ϕ_1, dag(newL), uniqueinds(ϕ_1, newL), uniqueinds(newL, ϕ_1); tags=("Left",)
  )
  ARⁿ², newr = ITensors.directsum(
    ϕ_2, dag(newR), uniqueinds(ϕ_2, newR), uniqueinds(newR, ϕ_2); tags=("Right",)
  )

  ###TODO remove assertions regarding expansion not exceeding maxdim ?
  @assert (dim(commonind(newL, S)) + old_linkdim) <= maxdim
  @assert dim(commonind(newL, S)) == dim(commonind(newR, S))
  @assert(dim(uniqueind(ϕ_1, newL)) + dim(uniqueind(newL, ϕ_1)) == dim(newl))
  @assert(dim(uniqueind(ϕ_2, newR)) + dim(uniqueind(newR, ϕ_2)) == dim(newr))
  @assert (old_linkdim + dim(commonind(newL, S))) <= maxdim
  @assert (old_linkdim + dim(commonind(newR, S))) <= maxdim
  @assert dim(newl) <= maxdim
  @assert dim(newr) <= maxdim

  ###zero-pad bond-tensor (the orthogonality center)
  C = ITensor(dag(newl)..., dag(newr)...)
  ψC = bondtensor
  for I in eachindex(ψC)
    v = ψC[I]
    if !iszero(v)
      C[I] = ψC[I]
    end
  end

  ###move orthogonality center back to site (should restore input orthogonality limits)
  if rlim == n2
    ψ[n2] = ARⁿ²
  elseif rlim > n2
    ψ[n2] = noprime(ARⁿ² * C)
  end

  if llim == n1
    ψ[n1] = ALⁿ¹
  elseif llim < n1
    ψ[n1] = noprime(ALⁿ¹ * C)
  end

  return nothing
end

function _subspace_expand_core(
  centerwf::Vector{ITensor}, env, NL, NR; maxdim, cutoff, kwargs...
)
  ϕ = ITensor(1.0)
  for atensor in centerwf
    ϕ *= atensor
  end
  return _subspace_expand_core(ϕ, env, NL, NR; maxdim, cutoff, kwargs...)
end

function _subspace_expand_core(ϕ::ITensor, env, NL, NR; maxdim, cutoff, kwargs...)
  ϕH = noprime(env(ϕ))   #add noprime?
  ϕH = NL * ϕH * NR
  if norm(ϕH) == 0.0
    return false, false, false, false
  end
  U, S, V = svd(
    ϕH,
    commoninds(ϕH, NL);
    maxdim,
    cutoff,
    use_relative_cutoff=false,
    use_absolute_cutoff=true,
    kwargs...,
  )

  @assert dim(commonind(U, S)) <= maxdim

  NL *= dag(U)
  NR *= dag(V)
  return NL, S, NR, true
end
