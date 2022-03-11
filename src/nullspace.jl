
ITensors.itensor(A::ITensor) = A

# Insert missing diagonal blocks
function insert_diag_blocks!(T::Tensor)
  for b in eachdiagblock(T)
    blockT = blockview(T, b)
    if isnothing(blockT)
      # Block was not found in the list, insert it
      insertblock!(T, b)
    end
  end
end
insert_diag_blocks!(T::ITensor) = insert_diag_blocks!(tensor(T))

# Reshape into an order-2 ITensor
matricize(T::ITensor, inds::Index...) = matricize(T, inds)

function matricize(T::ITensor, inds)
  left_inds = commoninds(T, inds)
  right_inds = uniqueinds(T, inds)
  return matricize(T, left_inds, right_inds)
end

function matricize(T::ITensor, left_inds, right_inds)
  CL = combiner(left_inds; dir=ITensors.Out, tags="CL")
  CR = combiner(right_inds; dir=ITensors.In, tags="CR")
  M = (T * CL) * CR
  return M, CL, CR
end

function setdims(t::NTuple{N,Pair{QN,Int}}, dims::NTuple{N,Int}) where {N}
  return first.(t) .=> dims
end

# XXX: generalize this function
function _getindex(T::DenseTensor{ElT,N}, I1::Colon, I2::UnitRange{Int64}) where {ElT,N}
  A = array(T)[I1, I2]
  return tensor(Dense(vec(A)), setdims(inds(T), size(A)))
end

function getblock(i::Index, n::Integer)
  return ITensors.space(i)[n]
end

# Make `Pair{QN,Int}` act like a regular `dim`
NDTensors.dim(qnv::Pair{QN,Int}) = last(qnv)

Base.:*(qnv::Pair{QN,Int}, d::ITensors.Arrow) = qn(qnv) * d => dim(qnv)

function getblock_preserve_qns(T::Tensor, b::Block)
  # TODO: make `T[b]` preserve QNs
  Tb = T[b]
  indsTb = getblock.(inds(T), Tuple(b)) .* dir.(inds(T))
  return ITensors.setinds(Tb, indsTb)
end

function blocksparsetensor(blocks::Dict{B,TB}) where {B,TB}
  b1, Tb1 = first(pairs(blocks))
  N = length(b1)
  indstypes = typeof.(inds(Tb1))
  blocktype = eltype(Tb1)
  indsT = getindex.(indstypes)
  # Determine the indices from the blocks
  for (b, Tb) in pairs(blocks)
    indsTb = inds(Tb)
    for n in 1:N
      bn = b[n]
      indsTn = indsT[n]
      if bn > length(indsTn)
        resize!(indsTn, bn)
      end
      indsTn[bn] = indsTb[n]
    end
  end
  T = BlockSparseTensor(blocktype, indsT)
  for (b, Tb) in pairs(blocks)
    if !isempty(Tb)
      T[b] = Tb
    end
  end
  return T
end

function _nullspace_hermitian(M::Tensor; atol::Real=0.0)
  tol = atol
  # Insert any missing diagonal blocks
  insert_diag_blocks!(M)
  #D, U = eigen(Hermitian(M))
  Dᵢₜ, Uᵢₜ = eigen(itensor(M); ishermitian=true)
  D = tensor(Dᵢₜ)
  U = tensor(Uᵢₜ)
  nullspace_blocks = Dict()
  for bU in nzblocks(U)
    bM = Block(bU[1], bU[1])
    bD = Block(bU[2], bU[2])
    # Assume sorted from largest to smallest
    indstart = sum(d -> abs(d) .> tol, storage(D[bD])) + 1
    Ub = getblock_preserve_qns(U, bU)
    indstop = lastindex(Ub, 2)
    Nb = _getindex(Ub, :, indstart:indstop)
    nullspace_blocks[bU] = Nb
  end
  return blocksparsetensor(nullspace_blocks)
end

function LinearAlgebra.nullspace(M::Hermitian{<:Number,<:Tensor}; kwargs...)
  return _nullspace_hermitian(parent(M); kwargs...)
end

function LinearAlgebra.nullspace(::Order{2}, M::ITensor, left_inds, right_inds; kwargs...)
  @assert order(M) == 2
  M² = prime(dag(M), right_inds) * M
  M² = permute(M², right_inds'..., right_inds...)
  M²ₜ = tensor(M²)
  Nₜ = nullspace(Hermitian(M²ₜ); kwargs...)
  indsN = (Index(ind(Nₜ, 1); dir=ITensors.In), Index(ind(Nₜ, 2); dir=ITensors.In))
  N = dag(itensor(ITensors.setinds(Nₜ, indsN)))
  # Make the index match the input index
  Ñ = replaceinds(N, (ind(N, 1),) => right_inds)
  return Ñ
end

function LinearAlgebra.nullspace(T::ITensor, is...; kwargs...)
  M, CL, CR = matricize(T, is...)
  @assert order(M) == 2
  cL = commoninds(M, CL)
  cR = commoninds(M, CR)
  N₂ = nullspace(Order(2), M, cL, cR; kwargs...)
  return N₂ * CR
end
