module SALTBase

using CatViews, StaticArrays
using LinearAlgebra
using Printf: @printf

export LinearSolverData

# Below, use Int instead of Int64 for compatibility with 32-bit systems (e.g., x86 in appveyor.yml)
const Float = typeof(0.0)  # use Float = Float128 for quadruple precision in the future
const CFloat = Complex{Float}

const AbsVec = AbstractVector
const AbsMat = AbstractMatrix
const AbsArr = AbstractArray

const VecBool = Vector{Bool}
const VecInt = Vector{Int}
const VecFloat = Vector{Float}
const VecComplex = Vector{CFloat}

# Note that Function is abstract type, unlike Bool, Int, Float, CFloat.  When we have
# functions f and g, [f, g] is Vector{Function}, but [f] or [g] is not.  I want to
# store even the latter as Vector{Function}.
const VecFun = Vector{Function}

const AbsVecInt = AbsVec{Int}
const AbsVecFloat = AbsVec{Float}
const AbsVecComplex = AbsVec{CFloat}

const AbsVecInteger = AbsVec{<:Integer}
const AbsVecReal = AbsVec{<:Real}
const AbsVecNumber = AbsVec{<:Number}
const AbsVecFunction = AbsVec{<:Function}

const MatFloat = Matrix{Float}
const MatComplex = Matrix{CFloat}

const AbsMatFloat = AbsMat{Float}
const AbsMatComplex = AbsMat{CFloat}

const AbsMatReal = AbsMat{<:Real}
const AbsMatNumber = AbsMat{<:Number}

# Each concrete SALT solver package (e.g., MaxwellSALT) must define a concrete subtype of
# LinearSolverData to store the information (e.g., the Maxwell operator A) needed by the
# specific linear Maxwell solver to use in the SALT solver package.
abstract type LinearSolverData end

# Each concrete SALT solver package (e.g., MaxwellSALT) must extend the following Base
# functions:
#
# - similar(lsd:LinearSolverData)
# - size(lsd::LinearSolverData)

# Each concrete SALT solver package (e.g., MaxwellSALT) must extend the following functions
# to be used in lasing.jl and nonlasing.jl:
function init_lsd!(lsd::LinearSolverData, ω::Number, ε::AbsVecNumber) end
function linsolve!(x::AbsVecComplex, lsd::LinearSolverData, b::AbsVecNumber) end
function linsolve_transpose!(x::AbsVecComplex, lsd::LinearSolverData, b::AbsVecNumber) end
function linapply!(b::AbsVecComplex, lsd::LinearSolverData, x::AbsVecNumber) end

include("base.jl")
include("gain.jl")
include("lasing.jl")
include("nonlasing.jl")
include("anderson.jl")
include("switching.jl")
include("simulation.jl")

end # module
