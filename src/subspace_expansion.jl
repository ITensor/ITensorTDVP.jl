
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
      inds = (ha == 1 ? (b,b+Δ) : (b+Δ,b))
      _=subspace_expansion!(ψ,PH,(ψ.llim,ψ.rlim),inds;maxdim, cutoff, atol=atol, kwargs...
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
  #@assert n1 + 1 == n2 || n1 -1 == n2
  PH.nsite=2
  if llim == n1
    @assert rlim == n2+1
    U,S,V=svd(ψ[n2],uniqueinds(ψ[n2],ψ[n1]);maxdim=maxdim, cutoff=0., kwargs...)  ##lookup svd interface again
    ϕ_1=ψ[n1]
    ϕ_2=U
    bondtensor = S * V
    
  elseif rlim==n2
    @assert llim == n1-1
    U,S,V=svd(ψ[n1],uniqueinds(ψ[n1],ψ[n2]);maxdim=maxdim, cutoff=0., kwargs...)
    ϕ_1=U
    ϕ_2=ψ[n2]
    bondtensor = S * V
  end
  position!(PH,ψ,min(n1,n2))
  @show PH.lpos
  @show PH.rpos
  @show ψ.llim
  @show ψ.rlim
  @show llim, rlim
  @show n1,n2
  
  
  #orthogonalize(ψ,n1)
  linkind_l=commonind(ϕ_1,bondtensor)
  linkind_r=commonind(ϕ_2,bondtensor)
  
  NL=nullspace(ϕ_1,linkind_l;atol=1e-9)
  NR=nullspace(ϕ_2,linkind_r;atol=1e-9)

  #@show norm(NL)
  #@show norm(NR)
  ###this is a crucial decision, justified for MPS, but will fail for rank-1 trees with physical DOFs on leafs
  ###vanishing norm should trigger a one-sided subspace expansion
  if norm(NL)==0.0 || norm(NR)==0.
    return bondtensor
  end
  
  ϕ=ϕ_1 * bondtensor * ϕ_2
  #ψ[n1]=prime(ψ[n1],linkind)
  
  
  newL,S,newR,success=_subspace_expand_core(ϕ,PH,NL,NR,;maxdim, cutoff, atol=1e-2, kwargs...)
  if success == false
    return nothing
  end
  ALⁿ¹, newl = ITensors.directsum(
    ϕ_1, dag(newL), uniqueinds(ϕ_1, newL), uniqueinds(newL, ϕ_1); tags=("Left",)
  )
  ARⁿ², newr = ITensors.directsum(
    ϕ_2, dag(newR), uniqueinds(ϕ_2, newR), uniqueinds(newR, ϕ_2); tags=("Right",)
  )
  #C,newlr=ITensors.directsum(bondtensor => (prime(linkind_l,1), linkind_r), nullbond => (uniqueinds(newL, ϕ_1),  uniqueinds(newR, ϕ_2)),tags=("Left","Right"))  
  C = ITensor(dag(newl)..., dag(newr)...)
  ψC = permute(bondtensor, linkind_l, linkind_r)
  for I in eachindex(ψC)
    v = ψC[I]
    if !iszero(v)
      C[I] = ψC[I]
    end
  end
  println("before")
  @show inds(ψ[n1])
  @show inds(ψ[n2])
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
  println("after")
  
  @show inds(ψ[n1])
  @show inds(ψ[n2])
  
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
  println("core")
  @show inds(ϕ)
  #@show env
  ϕH = noprime(env(ϕ))   #add noprime?
  ϕH = NL * ϕH * NR
  @show norm(ϕH)
  if norm(ϕH) == 0.0
    return false,false,false,false
  end
  U,S,V=svd(ϕH,commoninds(ϕH,NL);maxdim=maxdim, cutoff=cutoff, kwargs...)
  #@show inds(U)
  #@show inds(S)
  #@show inds(NL)
  
  NL *= dag(U)
  NR *= dag(V)
  return NL,S,NR,true
end