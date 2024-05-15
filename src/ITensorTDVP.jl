module ITensorTDVP
export TimeDependentSum, basis_extend, dmrg_x, linsolve, tdvp, to_vec
include("ITensorsExtensions.jl")
using .ITensorsExtensions: to_vec
include("applyexp.jl")
include("defaults.jl")
include("update_observer.jl")
include("timedependentsum.jl")
include("tdvporder.jl")
include("sweep_update.jl")
include("alternating_update.jl")
include("tdvp.jl")
include("dmrg.jl")
include("dmrg_x.jl")
include("reducedcontractproblem.jl")
include("contract.jl")
include("reducedconstantterm.jl")
include("reducedlinearproblem.jl")
include("linsolve.jl")
include("basis_extend.jl")
using PackageExtensionCompat: @require_extensions
function __init__()
  @require_extensions
end
end