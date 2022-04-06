using ITensors: AbstractProjMPO, ProjMPO, ProjMPOSum, ProjMPS, ProjMPO_MPS, DiskProjMPO

set_nsite!(::AbstractProjMPO, nsite) = error("Not implemented")

function set_nsite!(P::ProjMPO, nsite)
  P.nsite = nsite
  return P
end

function set_nsite!(Ps::ProjMPOSum, nsite)
  for P in Ps.pm
    set_nsite!(P, nsite)
  end
  return Ps
end

function set_nsite!(P::ProjMPS, nsite)
  P.nsite = nsite
  return P
end

function set_nsite!(Ps::ProjMPO_MPS, nsite)
  set_nsite!(Ps.PH, nsite)
  for P in Ps.pm
    set_nsite!(P, nsite)
  end
  return Ps
end

function set_nsite!(P::DiskProjMPO, nsite)
  P.nsite = nsite
  return P
end
