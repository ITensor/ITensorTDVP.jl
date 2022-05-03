#
# Possible improvements
#  - allow a maxdim argument to be passed to `extend`
#    and through `basis_extend`
#  - current behavior is letting bond dimension get too
#    big when used in imaginary time evolution
#  - come up with better names:
#    > should `basis_extend` be called `krylov_extend`?
#    > should `extend` be called `basis_extend`?
#  - Use (1-tau*H)|psi> to generate "Krylov" vectors
#    instead of H|psi>. Needed?
#

"""
Given an MPS psi and a collection of MPS phis,
returns an MPS which is equal to psi
(has fidelity 1.0 with psi) but whose MPS basis
is extended to contain a portion of the basis of
the phis that is orthogonal to the MPS basis of psi.
"""
function extend(psi::MPS, phis::Vector{MPS}; kwargs...)
  cutoff = get(kwargs, :cutoff, 1E-14)
  N = length(psi)
  psi = copy(psi)
  phis = copy(phis)

  orthogonalize!(psi, N)
  for phi in phis
    orthogonalize!(phi, N)
  end

  s = siteinds(psi)

  for j in reverse(2:N)
    # SVD psi[j] to compute B
    linds = (s[j - 1], linkind(psi, j - 1))
    _, S, B = svd(psi[j], linds; righttags="bψ_$j,Link")
    rinds = uniqueinds(B, S)

    # Make projector
    Id = ITensor(1.0)
    for r in rinds
      Id *= delta(r', dag(r))
    end
    P = Id - prime(B, rinds) * dag(B)

    # Sum phi density matrices
    rho = ITensor()
    for phi in phis
      rho += prime(phi[j], rinds) * dag(phi[j])
    end
    rho /= tr(rho)

    # Apply projector
    PrhoP = apply(apply(P, rho), P)

    if norm(PrhoP) > 1E-12
      # Diagonalize projected density matrix PrhoP
      # to compute Bphi, which spans part of right basis 
      # of phis which is orthogonal to right basis of psi
      D, Bphi = eigen(PrhoP; cutoff, ishermitian=true, righttags="bϕ_$j,Link")

      ## Test Bphi is ortho to B
      #O = Bphi*B
      #if norm(O) > 1E-10
      #  @show norm(O)
      #  error("Non-zero overlap of extended basis with original basis")
      #end

      # Form direct sum of B and Bphi over left index
      bψ = commonind(B, S)
      bϕ = commonind(Bphi, D)
      bx = Index(dim(bψ) + dim(bϕ), "bx_$(j-1)")
      D1, D2 = ITensors.directsum_itensors(bψ, bϕ, bx)
      Bx = D1 * B + D2 * Bphi
    else
      Bx = B
    end

    # Shift ortho center one site left using dag(Bx)
    # and replace tensor at site j with Bx
    psi[j - 1] = psi[j - 1] * (psi[j] * dag(Bx))
    psi[j] = Bx
    for phi in phis
      phi[j - 1] = phi[j - 1] * (phi[j] * dag(Bx))
      phi[j] = Bx
    end
  end

  return psi
end

function basis_extend(psi::MPS, H::MPO; kwargs...)
  kdim = get(kwargs, :extension_krylovdim, 2)
  maxdim = 1 + maxlinkdim(psi)

  phis = Vector{MPS}(undef, kdim)
  for k in 1:kdim
    prev = k == 1 ? psi : phis[k - 1]
    phis[k] = apply(H, prev; maxdim)
    normalize!(phis[k])
  end

  cutoff = get(kwargs, :extension_cutoff, 1E-8)
  psix = extend(psi, phis; cutoff)
  return psix
end
