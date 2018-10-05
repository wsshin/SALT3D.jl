using SALTBase
using Test

# Define a concrete subtype of LinearSolverData for the test purspose.
struct DefaultLSD <: LinearSolverData end
Base.similar(::DefaultLSD) = DefaultLSD()
Base.size(::DefaultLSD) = (0,0)

include("base.jl")
include("gain.jl")
include("lasing.jl")
