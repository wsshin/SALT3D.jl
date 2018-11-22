# Define variables used throughout the package and functions to initialize them.

export GainProfile
export gen_γ, gen_γ′, gen_abs2γ, gen_abs2γ′, hole_burning!

# Below, allow vectors and matrices to be PETSc ones if their sizes are 3×(# of grid points).
# If their sizes are the number of modes, keep them Julia vectors.

# About avoiding overtyping container-type variables:
#
# Below, for read-only container-type variables, I specify neither the container type nor
# eltype concretely.  [What are the examples of read-only container-type variables?]
#
# However, for variables into which I write, I don't specify the container type concretely
# (so that I can allow various container type, such as Julia vectors and PETSc vectors), but
# I do specify the eltype, because otherwise I get InexactError, which complains about
# incompatible element type (e.g., when attempting to store ComplexF64 in Float64).  [I'm
# not sure what I intended to mean here.]
#
# For this reason, I have two aliases for abstract container types: one with a concrete
# eltype and the other with an abstract eltype.  For example, I have AbsVecComplex with a
# concrete eltype of ComplexF64, and also AbsVecNumber with an abstract eltype of Number.
#
# Consider SATLParam below.  εc is typed differently when it is used as a field and as an
# argument of a constructor.
# - The field type VC of εc, into which I write, is a subtype of AbsVecComplex, whose eltype
# is the concrete ComplexF64.
# - On the other hand, the argument type of εc in the inner constructor is not AbsVecComplex,
# but AbsVecNumber, because even if its element is not typed ComplexF64, it is converted to
# a vector with ComplexF64 when writing in the field εc of GainProfile.  (Note that the
# argument εc of the inner constructor is copied to the field εc, so it is read-only.)
# - [I think this item is outdated, because the current outer constructor take εc of
# AbsVecNumber type.]  Still, in the outer constructor, the argument type of εc is not
# AbsVecNumber but AbsVecComplex.  This is because we want to call the inner constructor
# while specifying the type of εc concretely for type stability, and the eltype of the field
# εc is not necessarily the same as the eltype of the argument εc.  Specifically, if we type
# εc as AbsVecNumber at the signature of the outer constructor, then we can pass a PETSc
# vector of Float64 to it, and I would want to use a PETSc vector of ComplexF64 for VC.  How
# would I be able to deduce such a type?  Upon calling the outer constructor, the concrete
# type of the argument εc, i.e., PETSc vector of Float64, is decided, but still I cannot
# deduce ComplexF64 from Float64...  [Well, technically this is doable even if the eltype of
# the argument εc was Int64, by complex(float(eltype(εc))).  Wait, maybe PETSc's Float64 is
# not Julia's Float64, so PETSc.jl must overload float() and complex() functions for this
# capability?  In that case, even if such deduction was possible in PETSc.jl, similar
# deduction might not be possible for other vector types, so it's safe to use a concrete
# eltype in the outer constructor?]
#
# The same principle applies to writing functions as well as types.  If the function changes
# the contents of some variable, that variable is where I write, so its eltype must be
# specified to avoid InexactError.  For example, in create_A!, the argument A is where the
# output is stored, so it had better have ComplexF64 as the eltype, so I type it as
# AbsMatComplex.  On the other hand, the argument CC is read-only, so it is typed AbsMatNumber.

# Generate the default gain curve of SALT, which describes two-level atoms.
gen_γ(ω₀::Real, γperp::Real) = ω::Number -> γperp / (ω - ω₀ + im * γperp)  # scalar
gen_γ′(ω₀::Real, γperp::Real) = ω::Number -> -γperp / (ω - ω₀ + im * γperp)^2  # scalar

# Below, we define the formula for |γ|².  This can be easily calculated from γ, but the
# users may accidentally evaluate γ(ω) for complex ω and take its squared absolute value.
# That will give a wrong result, because the imaginary part of ω will be added to iγ in the
# denominator when calculating the absolute value.  To prevent this unfortunate accident
# from happening, we define a formula for |γ|² that accepts only real ω and ask the users to
# use it.
gen_abs2γ(ω₀::Real, γperp::Real) = ω::Real -> γperp^2 / ((ω - ω₀)^2 + γperp^2)  # scalar; note ω is real for this
gen_abs2γ′(ω₀::Real, γperp::Real) = ω::Real -> -2γperp^2 * (ω - ω₀) / ((ω - ω₀)^2 + γperp^2)^2  # scalar; note ω is real for this

# Parameters defining the SALT problem
# Consider including CC to gp, if I am really going to use ω₀ for PML for all modes.
mutable struct GainProfile{VF<:AbsVecFloat}  # VF can be PETSc vectors
    gain::VecFun  # gain curve
    gain′::VecFun  # derivative of gain curve
    abs2gain::VecFun  # squared absolute value of gain curve
    abs2gain′::VecFun  # derivative of squared absolute value of gain curve
    D₀::VF  # pump strength
    wt::VecFun  # function that calculates contribution of D₀ to kth atomic class; D₀ₖ = wt.(D₀)
    function GainProfile{VF}(gain::AbsVecFunction,
                             gain′::AbsVecFunction,
                             abs2gain::AbsVecFunction,
                             abs2gain′::AbsVecFunction,
                             D₀::AbsVecReal,
                             wt::AbsVecFunction) where {VF<:AbsVecFloat}

        K = length(wt)  # number of atomic classes
        length(gain)==length(gain′)==length(abs2gain)==length(abs2gain′)==K ||
            throw(ArgumentError("length(gain) = $(length(gain)), length(gain′) = $(length(gain′)), " *
                                "length(abs2gain) = $(length(abs2gain)), length(abs2gain′) = $(length(abs2gain′)) must be the same as length(wt) = $(length(wt))."))

        return new(gain, gain′, abs2gain, abs2gain′, D₀, wt)
    end
end

function GainProfile(gain::AbsVecFunction, gain′::AbsVecFunction, abs2gain::AbsVecFunction, abs2gain′::AbsVecFunction, D₀::AbsVecReal,
                     wt::AbsVecFunction=(K=length(gain); [(d::Real->d/K) for k=1:K]))  # default wt: even distribution
    D₀_new = similar(D₀,Float)
    copyto!(D₀_new, D₀)

    return GainProfile{typeof(D₀_new)}(gain, gain′, abs2gain, abs2gain′, D₀_new, wt)
end

# Single atomic class
GainProfile(gain::Function, gain′::Function, abs2gain::Function, abs2gain′::Function, D₀::AbsVecReal) =
    GainProfile([gain], [gain′], [abs2gain], [abs2gain′], D₀)
GainProfile(ω, vtemp::AbsVec) =  # vtemp: template vector with N entries
    GainProfile(gain, gain′, abs2gain, abs2gain′, similar(vtemp,Float))
GainProfile(gain::Function, gain′::Function, abs2gain::Function, abs2gain′::Function, N::Integer) =
    GainProfile(gain, gain′, abs2gain, abs2gain′, VecFloat(undef,N))

#= Convenience constructors with Lorentzian parameters =#
# Multiple atomic classes
function GainProfile(ω₀::AbsVecReal, γperp::AbsVecReal, D₀::AbsVecReal,
                     wt::AbsVecReal=(K=length(ω₀); fill(1.0/K,K)))  # default wt: even distribution
    K = length(wt)
    length(ω₀)==length(γperp)==K ||
        throw(ArgumentError("length(ω₀) = $(length(ω₀)), length(γperp) = $(length(γperp)), " *
                            "length(wt) = $(length(wt)) must be the same."))

    γ = [gen_γ(float(ω₀[k]), float(γperp[k])) for k = 1:K]
    γ′ = [gen_γ′(float(ω₀[k]), float(γperp[k])) for k = 1:K]
    abs2γ = [gen_abs2γ(float(ω₀[k]), float(γperp[k])) for k = 1:K]
    abs2γ′ = [gen_abs2γ′(float(ω₀[k]), float(γperp[k])) for k = 1:K]
    wtfun = [(d::Real->float(wt[k]*d)) for k = 1:K]

    return GainProfile(γ, γ′, abs2γ, abs2γ′, D₀, wtfun)
end
GainProfile(ω₀::AbsVecReal, γperp::AbsVecReal, N::Integer, wt::AbsVecReal=(K=length(ω₀); fill(1.0/K,K))) =  # default wt: even distribution
    GainProfile(ω₀, γperp, VecFloat(undef,N), wt)
GainProfile(ω₀::AbsVecReal, γperp::Real, N::Integer, wt::AbsVecReal=(K=length(ω₀); fill(1.0/K,K))) =  # default wt: even distribution
    GainProfile(ω₀, fill(γperp,length(ω₀)), VecFloat(undef,N), wt)

# Single atomic class
GainProfile(ω₀::Real, γperp::Real, D₀::AbsVecReal) = GainProfile(gen_γ(ω₀,γperp), gen_γ′(ω₀,γperp), gen_abs2γ(ω₀,γperp), gen_abs2γ′(ω₀,γperp), D₀)
GainProfile(ω₀::Real, γperp::Real, vtemp::AbsVec) =  GainProfile(gen_γ(ω₀,γperp), gen_γ′(ω₀,γperp), gen_abs2γ(ω₀,γperp), gen_abs2γ′(ω₀,γperp), vtemp::AbsVec)
GainProfile(ω₀::Real, γperp::Real, N::Integer) = GainProfile(gen_γ(ω₀,γperp), gen_γ′(ω₀,γperp), gen_abs2γ(ω₀,γperp), gen_abs2γ′(ω₀,γperp), N::Integer)

Base.length(gp::GainProfile) = length(gp.wt)  # number of atomic classes

# Evaluate 1 + hole-burning term = 1 + ∑|γaψ|².
function hole_burning!(hb::AbsVecNumber,  # output
                       abs2gain::Function,  # function that takes ω and produces |γ(ω)|²
                       ω::AbsVecReal,  # vector of lasing frequencies
                       a²::AbsVecReal,  # vector of squared amplitudes of unnormalized eigenmodes
                       abs2ψ::AbsVec{<:AbsVecReal})  # vector of absolute squares of normalized eigenmodes
    hb .= 1  # initialize
    for m = 1:length(a²)
        if a²[m] ≠ 0
            # @info "a²[$m] = $(a²[m]), ‖ψ[$m]‖ = $(norm(ψ[m]))"
            hb .+=  (abs2gain(ω[m]) * a²[m]) .* abs2ψ[m]
        end
    end
    # @info "‖hb‖ = $(norm(hb))"

    return nothing
end
