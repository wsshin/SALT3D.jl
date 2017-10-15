# Define variables used throughout the package and functions to initialize them.

export SALTParam
export gain, gain′, hole_burning!, create_A!

# Below, allow vectors and matrices to be PETSc ones if their sizes are 3×(# of grid points).
# If their sizes are the number of modes, keep them Julia vectors.

# About avoiding overtyping:
#
# Below, for read-only variables, I specify neither the container type nor eltype.  However,
# for variables that I write in, I don't specify the container type (so that I can allow
# various container type, such as Julia vectors and PETSc vectors), but I do specify the
# eltype, because otherwise I get InexactError, which complains about incompatible element
# type (e.g., when attempting to store Complex128 in Float64).
#
# For example, in SATLParam,
# - The field type VC of εc, where I write, is a subtype of AbsVecComplex, whose eltype is
# the concrete Complex128.
# - On the other hand, the argument type of εc in the inner constructor is not AbsVecComplex,
# but AbsVecNumber, because even if the element is not ::Complex, it is converted to
# Complex128 when writing in the field εc of SALTParam.
# - Still, in the outer constructor, the argument type of εc is not AbsVecNumber, but
# AbsVecComplex, because from a subtype of AbsVecNumber, it is not easy to deduce the right
# type parameter VC of the inner constructor to call.  For example, if εc is a PETSc vector
# of Float64, I would like to use a PETSc vector of Complex128 for VC, but how can I deduce
# such a type?  Even if such deduction is possible in PETSc.jl, similar deduction may not be
# possible for other vector types.)
#
# The same principle applies to writing functions as well as types.  If the function changes
# the contents of some variable, that variable is where I write, so its eltype must be
# specified to avoid InexactError.  For example, in create_A!, the argument A is where the
# output is stored, so it had better having Complex128 as the eltype, so I type it as
# AbsMatComplex.  On the other hand, the argument CC is read-only, so it is typed AbsMatNumber.

# Parameters defining the SALT problem
# Consider including CC to param, if I am really going to use ωₐ for PML for all modes.
mutable struct SALTParam{VC<:AbsVecComplex,VF<:AbsVecFloat}  # VC, VF can be PETSc vectors
    ωₐ::Float  # atomic transition angular frequency
    γ⟂::Float  # relaxation rate of polarization
    εc::VC  # permittivity of cold cavity
    D₀::VF  # pump strength
    function SALTParam{VC,VF}(ωₐ::Real,
                              γ⟂::Real,
                              εc::AbsVecNumber,
                              D₀::AbsVecReal) where {VC<:AbsVecComplex,VF<:AbsVecFloat}
        length(εc) == length(D₀) ||
            throw(ArgumentError("legnth(εc) == $(length(εc)) and length(D₀) == $(length(D₀)) must be the same"))

        return new(ωₐ, γ⟂, εc, D₀)
    end
end

# # The following avoids copying εc and D₀.
# SALTParam(ωₐ::Real, γ⟂::Real, εc::VC, D₀::VF) where {VC<:AbsVecComplex,VF<:AbsVecFloat} =
#     SALTParam{VC,VF}(ωₐ, γ⟂, εc, D₀)

# The following copies εc and D₀.
function SALTParam(ωₐ::Real, γ⟂::Real, εc::AbsVecNumber, D₀::AbsVecReal)
    εc_new = similar(εc,CFloat)
    copy!(εc_new, εc)

    D₀_new = similar(D₀,Float)
    copy!(D₀_new, D₀)

    return SALTParam{typeof(εc_new), typeof(D₀_new)}(ωₐ, γ⟂, εc_new, D₀_new)
end

# To do: check if the following works for vtemp of PETSc vector type.
SALTParam(vtemp::AbsVec) =  # template vector with N entries
    SALTParam(0, 0, similar(vtemp,CFloat), similar(vtemp,Float))
SALTParam(N::Integer) = SALTParam(Vector{Float}(N))


gain(ω::Number, ωₐ::Number, γ⟂::Number) = γ⟂ / (ω - ωₐ + im * γ⟂)  # scalar
gain′(ω::Number, ωₐ::Number, γ⟂::Number) = -γ⟂ / (ω - ωₐ + im * γ⟂)^2  # scalar


# Evaluate the hole-burning term 1 + ∑a²|ψ|².
function hole_burning!(hb::AbsVecFloat,  # output
                       a²::AbsVecReal,  # vector of squared amplitudes of unnormalized eigenmodes
                       ψ::AbsVec{<:AbsVecNumber})  # vector of normalized eigenmodes
    hb .= 1  # initialize
    for m = 1:length(a²)
        if a²[m] > 0
            # info("a²[$m] = $(a²[m]), ‖ψ[$m]‖ = $(norm(ψ[m]))")
            hb .+=  a²[m] .* abs2.(ψ[m])
        end
    end
    # info("‖hb‖ = $(norm(hb))")

    return nothing
end


# Create ∇×∇× - ω² (ε + γ(ω) D).
function create_A!(A::AbsMatComplex,  # output; must have same nonzero entry pattern as CC
                   CC::AbsMatNumber,  # curl of curl (∇×∇×)
                   ω::Number,  # angular frequency
                   ε::AbsVecNumber)  # effective ε
    A .= CC  # initialize; works for sparse matrices with same nonzero entry pattern
    # info("‖CC‖₁ = $(norm(CC,1)), ω = $ω, ‖ε‖ = $(norm(ε))")
    for i = 1:length(ε)
        A[i,i] -= ω^2 * ε[i]
    end

    return nothing
end
