module SALT3D

using MaxwellFDM, GeometryPrimitives, CatViews, StaticArrays

# package code goes here
const Float = typeof(0.0)  # use Float = Float128 for quadruple precision in the future
const CFloat = Complex{Float}

const AbsVec = AbstractVector
const AbsMat = AbstractMatrix
const AbsArr = AbstractArray

const VecBool = Vector{Bool}
const VecInt = Vector{Int}
const VecFloat = Vector{Float}
const VecComplex = Vector{CFloat}

const AbsVecInt = AbsVec{Int}
const AbsVecFloat = AbsVec{Float}
const AbsVecComplex = AbsVec{CFloat}

const AbsVecInteger = AbsVec{<:Integer}
const AbsVecReal = AbsVec{<:Real}
const AbsVecNumber = AbsVec{<:Number}

const MatFloat = Matrix{Float}
const MatComplex = Matrix{CFloat}

const AbsMatFloat = AbsMat{Float}
const AbsMatComplex = AbsMat{CFloat}

const AbsMatReal = AbsMat{<:Real}
const AbsMatNumber = AbsMat{<:Number}

include("base.jl")
include("salteq.jl")
include("lasing.jl")
include("nonlasing.jl")
include("gainobj.jl")
include("anderson.jl")
include("switching.jl")
include("simulation.jl")

end # module
