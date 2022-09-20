"""
#fields
- `maxtruncerr::Float64`: the maximum tuncation error
- `converged::Int`: 1 if converged, 0 if not, -1 if NA
"""
struct TDVPInfo
  maxtruncerr::Float64
  converged::Int
end
