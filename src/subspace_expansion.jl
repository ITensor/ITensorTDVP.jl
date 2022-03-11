
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
function subspace_expansion_sweep!(ψ::MPS,PH::Union{ProjMPO,ProjMPOSum};maxdim, cutoff, atol=1e-2, kwargs...)
  N = length(ψ)
  if !isortho(ψ) || orthocenter(ψ) != 1
    orthogonalize!(ψ, 1)
  end
  PH.nsite=2
  nsite=2
  position!(PH, ψ, 1)
  for (b, ha) in sweepnext(N; ncenter=2)
    println(b)
    orthogonalize!(ψ,b)
    position!(PH, ψ, b)
    
    if (ha == 1 && (b + nsite - 1 != N)) || (ha == 2 && b != 1)
      b1 = (ha == 1 ? b + 1 : b)
      Δ = (ha == 1 ? +1 : -1)
      println("")
      _=subspace_expansion!(ψ,PH,(ψ.llim,ψ.rlim),(b,b+Δ);maxdim, cutoff, atol=atol, kwargs...
      )
    end
  end
  return nothing
end

function subspace_expansion!(ψ::MPS,PH,lims::Tuple{Int,Int},b::Tuple{Int,Int};bondtensor=nothing,maxdim, cutoff, atol=1e-2, kwargs...)
  ##this should only work for the case where rlim-llim > 1
  ##not a valid MPS otherwise anyway (since bond matrix not part of MPS unless so defined like in VidalMPS struct)
  llim,rlim = lims
  n1, n2 = b
  @assert n1 + 1 == n2
  PH.nsite=2
  position!(PH,ψ,n1)
  linkind=commonind(ψ[n1],ψ[n2])
  @show ψ[n1]
  NL=nullspace(ψ[n1],linkind;atol=1e-6)
  NR=nullspace(ψ[n2],linkind;atol=1e-6)
  ϕ=ψ[n1]*ψ[n2]
  ψ[n1]=prime(ψ[n1],linkind)
  if isnothing(bondtensor)
    bondtensor=diagITensor(1.0,prime(linkind,1),linkind)
  end
  
  newL,S,newR=_subspace_expand_core(ϕ,PH,NL,NR,;maxdim, cutoff, atol=1e-2, kwargs...)
  nullbond=diagITensor(0.0,uniqueinds(newL, ψ[n1]),uniqueinds(newR, ψ[n2]))


  ALⁿ¹, newl = ITensors.directsum(
    ψ[n1], dag(newL), uniqueinds(ψ[n1], newL), uniqueinds(newL, ψ[n1]); tags=("Left",)
  )
  ARⁿ², newr = ITensors.directsum(
    ψ.AR[n2], dag(newR), uniqueinds(ψ[n2], newR), uniqueinds(newR, ψ[n2]); tags=("Right",)
  )
  C,newlr=ITensors.directsum(bondtensor => (prime(linkind,1), linkind), nullbond => (uniqueinds(newL, ψ[n1]),  uniqueinds(newR, ψ[n2])),tags=("Left","Right"))
  
  if rlim==n2
    ψ[n2]=ARⁿ²
  elseif rlim>n2
    ψ[n2]=noprime(ARⁿ²*C)
  end

  if llim==n1
    ψ[n1]=ALⁿ¹
  elseif llim<n1
    ψ[n1]=noprime(ALⁿ¹*C)
  end
  
  return C
end 

function _subspace_expand_core(centerwf::Vector{ITensor}, env,NL,NR;maxdim, cutoff, atol=1e-2, kwargs...)
  ϕ = ITensor(1.0)
  for atensor in centerwf
    ϕ *= atensor
  end
  return _subspace_expand_core(ϕ, env, NL, NR;maxdim, cutoff, atol=1e-2, kwargs...)  
end

function _subspace_expand_core(ϕ::ITensor, env,NL,NR;maxdim, cutoff, atol=1e-2, kwargs...)
  ϕH = env*ϕ   #add noprime?
  ϕH = NL * ϕH * NR
  U,S,V=svd(ϕH,commoninds(ϕH,NL);maxdim=maxdim, cutoff=cutoff, kwargs...)
  NL *= dag(U)
  NR *= dag(V)
  return NL,S,NR
end