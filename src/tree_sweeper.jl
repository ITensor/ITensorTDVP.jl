using ITensorNetworks: AbstractEdge, AbstractDataGraph

# 
# Sweep step
# 

"""
  struct SweepStep{V}

Auxiliary object specifying a single local update step in a tree sweeping algorithm.
"""
struct SweepStep{V} # parametrize on position type?
  pos::Union{Vector{<:V},NamedDimEdge{V}}
  time_direction::Base.Ordering # definitely want this to be an integer
end

# field access
pos(st::SweepStep) = st.pos
nsite(st::SweepStep) = isa(pos(st), AbstractEdge) ? 0 : length(pos(st))
time_direction(st::SweepStep) = st.time_direction
time_prefactor(st::SweepStep) = time_prefactor(time_direction(st))
time_prefactor(::Base.ForwardOrdering) = 1
time_prefactor(::Base.ReverseOrdering) = -1

# utility
current_ortho(st::SweepStep) = current_ortho(typeof(pos(st)), st)
current_ortho(::Type{<:Vector{<:V}}, st::SweepStep{V}) where {V} = first(pos(st)) # not very clean...
current_ortho(::Type{NamedDimEdge{V}}, st::SweepStep{V}) where {V} = src(pos(st))

# 
# Abstract tree sweeper type
# 

abstract type AbstractTreeSweeper{V} end

# some general constructor utilities

function (::Type{SWT})(
  graph::AbstractDataGraph, args...; kwargs...
) where {SWT<:AbstractTreeSweeper}
  return SWT(underlying_graph(graph), args...; kwargs...)
end

function (::Type{SWT})(
  graph::NamedDimGraph{V};
  direction=Base.Forward,
  root_vertex::V=default_root_vertex(graph),
  reverse_step=false,
) where {SWT<:AbstractTreeSweeper,V}
  leaf_vertex = first(post_order_dfs_vertices(graph, root_vertex))
  return SWT{V}(graph, root_vertex, leaf_vertex, direction, reverse_step)
end

direction(s::AbstractTreeSweeper) = s.direction
do_reverse_step(s::AbstractTreeSweeper) = s.reverse_step

# functionalities required from every concrete tree sweeper

isforward(s::AbstractTreeSweeper) = isforward(direction(s))
isreverse(s::AbstractTreeSweeper) = isreverse(direction(s))

sweep_steps(s::AbstractTreeSweeper) = sweep_steps(direction(s), s)

function is_forward_done(s::AbstractTreeSweeper, step::SweepStep)
  return is_forward_done(direction(s), s, step)
end
is_forward_done(::Base.ReverseOrdering, s::AbstractTreeSweeper, step::SweepStep) = false

function is_reverse_done(s::AbstractTreeSweeper, step::SweepStep)
  return is_reverse_done(direction(s), s, step)
end
is_reverse_done(::Base.ForwardOrdering, s::AbstractTreeSweeper, step::SweepStep) = false

function is_half_sweep_done(s::AbstractTreeSweeper, step::SweepStep)
  return is_forward_done(s, step) || is_reverse_done(s, step)
end

# 
# One-site sweeper for trees
# 

"""
    OneSiteTreeSweeper{V} <: AbstractTreeSweeper{V}

One-site sweeper for tree graphs.
"""
struct OneSiteTreeSweeper{V} <: AbstractTreeSweeper{V}
  graph::NamedDimGraph{V}
  root_vertex::V
  leaf_vertex::V
  direction::Base.Ordering
  reverse_step::Bool
  function OneSiteTreeSweeper{V}(
    graph::NamedDimGraph{V},
    root_vertex::V,
    leaf_vertex::V,
    direction::Base.Ordering,
    reverse_step::Bool,
  ) where {V}
    return if leaf_vertex != first(post_order_dfs_vertices(graph, root_vertex))
      error("Inconsistent tree sweeper.")
    else
      new(graph, root_vertex, leaf_vertex, direction, reverse_step)
    end
  end
end

function sweep_steps(::Base.ForwardOrdering, s::OneSiteTreeSweeper{V}) where {V}
  edges = post_order_dfs_edges(s.graph, s.root_vertex)
  steps = SweepStep{V}[]
  for e in edges
    push!(steps, SweepStep{V}([src(e)], Base.Forward))
    do_reverse_step(s) && push!(steps, SweepStep{V}(e, Base.Reverse))
  end
  push!(steps, SweepStep{V}([s.root_vertex], Base.Forward))
  return steps
end

# is this even necessary?
function sweep_steps(::Base.ReverseOrdering, s::OneSiteTreeSweeper{V}) where {V}
  edges = reverse.(reverse(post_order_dfs_edges(s.graph, s.root_vertex)))
  steps = [SweepStep{V}([s.root_vertex], Base.Forward)]
  for e in edges
    do_reverse_step(s) && push!(steps, SweepStep{V}(e, Base.Reverse))
    push!(steps, SweepStep{V}([dst(e)], Base.Forward))
  end
  return steps
end

function is_forward_done(::Base.ForwardOrdering, s::OneSiteTreeSweeper, step::SweepStep)
  return nsite(step) == 1 && only(step.pos) == s.root_vertex
end

function is_reverse_done(::Base.ReverseOrdering, s::OneSiteTreeSweeper, step::SweepStep)
  return nsite(step) == 1 && only(step.pos) == s.leaf_vertex
end

# 
# Two-site sweeper for trees
# 

"""
    TwoSiteTreeSweeper{V} <: TreeSweeper{V}

Two-site sweeper for tree graphs.
"""
struct TwoSiteTreeSweeper{V} <: AbstractTreeSweeper{V}
  graph::NamedDimGraph{V}
  root_vertex::V
  leaf_vertex::V
  direction::Base.Ordering
  reverse_step::Bool
  function TwoSiteTreeSweeper{V}(
    graph::NamedDimGraph{V},
    root_vertex::V,
    leaf_vertex::V,
    direction::Base.Ordering,
    reverse_step::Bool,
  ) where {V}
    return if leaf_vertex != first(post_order_dfs_vertices(graph, root_vertex))
      error("Inconsistent sweeper.")
    else
      new(graph, root_vertex, leaf_vertex, direction, reverse_step)
    end
  end
end

function sweep_steps(::Base.ForwardOrdering, s::TwoSiteTreeSweeper{V}) where {V}
  edges = post_order_dfs_edges(s.graph, s.root_vertex)
  steps = SweepStep{V}[]
  for e in edges[1:(end - 1)]
    push!(steps, SweepStep{V}([src(e), dst(e)], Base.Forward))
    do_reverse_step(s) && push!(steps, SweepStep{V}([dst(e)], Base.Reverse))
  end
  push!(steps, SweepStep{V}([src(edges[end]), dst(edges[end])], Base.Forward))
  return steps
end

# is this even necessary?
function sweep_steps(::Base.ReverseOrdering, s::TwoSiteTreeSweeper{V}) where {V}
  edges = reverse.(reverse(post_order_dfs_edges(s.graph, s.root_vertex)))
  steps = [SweepStep{V}([src(edges[1]), dst(edges[1])], Base.Forward)]
  for e in edges[2:end]
    do_reverse_step(s) && push!(steps, SweepStep{V}([src(e)], Base.Reverse))
    push!(steps, SweepStep{V}([src(e), dst(e)], Base.Forward))
  end
  return steps
end

function is_forward_done(::Base.ForwardOrdering, s::TwoSiteTreeSweeper, step::SweepStep)
  return nsite(step) == 2 && last(step.pos) == s.root_vertex
end

function is_reverse_done(::Base.ReverseOrdering, s::TwoSiteTreeSweeper, step::SweepStep)
  return nsite(step) == 2 && last(step.pos) == s.leaf_vertex
end
