# Unlike lasing equations, nonlasing equations are not coupled between nonlasing modes, so
# all the functions are for a single nonlasing mode, and therefore they always take a mode
# index.

export NonlasingSol, NonlasingVar

# Solutions to the nonlasing equation.
mutable struct NonlasingSol{VC<:AbsVecComplex}  # VC can be PETSc vector
    ω::VecComplex  # M complex numbers: frequencies of modes (M = # of nonlasing modes)
    ψ::Vector{VC}  # M complex vectors: normalized modes
    iₐ::VecInt  # M integers: row indices where amplitudes are measured
    vtemp::VC  # temporary storage for N complex numbers; note its contents can change at any point
    activated::VecBool  # activated[m] == true if mode m is just activated; used in simulate!
    active::VecBool  # active[m] is true if mode m is activeive (i.e., nonlasing)
    m_active::VecInt  # vector of activeive (i.e., nonlasing) mode indices; collection of m such that active[m] == true
    function NonlasingSol{VC}(ω::AbsVecNumber,
                              ψ::AbsVec{<:AbsVecNumber},
                              iₐ::AbsVecInteger,
                              vtemp::AbsVecNumber) where {VC<:AbsVecComplex}
        # Test sizes.
        length(ω)==length(ψ)==length(iₐ) ||
            throw(ArgumentError("length(ω) = $(length(ω)), length(ψ) = $(length(ψ)),
                                 length(iₐ) = $(length(iₐ)) must be the same."))

        M = length(ψ)
        if M > 0
            N = length(ψ[1])
            for m = 2:M
                length(ψ[m]) == N ||
                    throw(ArgumentError("length(ψ[$m]) = $(length(ψ[m])) and length(ψ[1]) = $N must be the same."))
            end
        end

        return new(ω, ψ, iₐ, vtemp, fill(false,M), fill(true,M), VecInt(1:M))  # active[m]==true for all m: all modes are nonlasing
    end
end
NonlasingSol(ω::AbsVecNumber, ψ::AbsVec{VC}, iₐ::AbsVecInteger, vtemp::VC) where {VC<:AbsVecComplex} =
    NonlasingSol{VC}(ω, ψ, iₐ, vtemp)
NonlasingSol(ω::AbsVecNumber, Ψ::AbsMatComplex, iₐ::AbsVecInteger) =
    ((N,M) = size(Ψ); NonlasingSol(ω, [Ψ[:,m] for m = 1:M], iₐ, similar(Ψ,N)))

NonlasingSol(ω::AbsVecNumber, ψ::AbsVec{<:AbsVecComplex}) = NonlasingSol(ω, ψ, zeros(Int,length(ω)), similar(ψ[1]))  # iₐ is set here
NonlasingSol(ω::AbsVecNumber, Ψ::AbsMatComplex) = (M = length(ω); NonlasingSol(ω, [Ψ[:,m] for m = 1:M]))


# To do: check if the following works for vtemp of PETSc vector type.
NonlasingSol(vtemp::AbsVec, M::Integer) =  # vtemp has N entries
    NonlasingSol(zeros(CFloat,M), [similar(vtemp,CFloat).=0 for m = 1:M], zeros(Int,M), similar(vtemp,CFloat))
NonlasingSol(N::Integer, M::Integer) = NonlasingSol(VecFloat(undef,N), M)

Base.length(nlsol::NonlasingSol) = length(nlsol.ψ)

# Note that this function changes iₐ.  Therefore, this must not be called inside the
# iteration for finding the solution for a given pump strength, because iₐ must be kept the
# same for a given pump strength in order to solve the equations with the same normalization
# conditions.
function LinearAlgebra.normalize!(nlsol::NonlasingSol)
    for m = nlsol.m_active
        ψ = nlsol.ψ[m]
        iₐ = argmax(abs, ψ)
        nlsol.iₐ[m] = iₐ
        ψ ./= ψ[iₐ]  # make ψ[iₐ] = 1
    end

    return nothing
end

# nonlasing reduced bar: D
# nonlasing modal var: γ, γ′, εeff, A

# Nonlasing equation variables that are unique to each nonlasing mode
mutable struct NonlasingModalVar{LSD<:LinearSolverData,VC<:AbsVecComplex}  # VC can be PETSc matrix and vector
    lsd::LinearSolverData
    ∂f∂ω::VC
    inited::Bool
    function NonlasingModalVar{LSD,VC}(lsd::LSD, ∂f∂ω::AbsVecNumber) where {LSD<:LinearSolverData,VC<:AbsVecComplex}
        N = length(∂f∂ω)
        size(lsd) == (N,N) || throw(ArgumentError("Each entry of size(lsd) = $(size(lsd)) and length∂f∂ω) = $N must be the same."))

        return new(lsd, ∂f∂ω, false)
    end
end
NonlasingModalVar(lsd::LSD, ∂f∂ω::VC) where {LSD<:LinearSolverData,VC<:AbsVecComplex} =
    NonlasingModalVar{LSD,VC}(lsd, ∂f∂ω)

# To do: check if the following works for vtemp of PETSc vector type.
NonlasingModalVar(lsd_temp::LinearSolverData, vtemp::AbsVec) =   # vtemp has N entries
    NonlasingModalVar(similar(lsd_temp), similar(vtemp,CFloat))


# Collection of nonlasing equation variables
mutable struct NonlasingVar{LSD<:LinearSolverData,VC<:AbsVecComplex}
    mvar_vec::Vector{NonlasingModalVar{LSD,VC}}
end
NonlasingVar(lsd_temp::LinearSolverData, vtemp::AbsVec, M::Integer) = NonlasingVar([NonlasingModalVar(lsd_temp, vtemp) for m = 1:M])
NonlasingVar(lsd_temp::LinearSolverData, N::Integer, M::Integer) = NonlasingVar(lsd_temp, VecFloat(undef,N), M)


function init_nlvar!(nlvar::NonlasingVar, m::Integer, nlsol::NonlasingSol, D::AbsVec{<:AbsVecFloat}, gp::GainProfile, εc::AbsVecComplex)
    mvar = nlvar.mvar_vec[m]
    init_modal_var_impl!(mvar.lsd, mvar.∂f∂ω, nlsol.vtemp, nlsol.ω[m], nlsol.ψ[m], D, gp, εc)  # see lasing.jl
    mvar.inited = true

    return nothing
end

# Unlike norm_leq, this norm_nleq takes the mode index m, because nonlasing mode equations
# are uncoupled between nonlasing modes.  This asymmetry between norm_leq and norm_lneq may
# need to be fixed because it makes tracking code difficult.
function norm_nleq(m::Integer, nlsol::NonlasingSol, mvar_vec::AbsVec{<:NonlasingModalVar})
    mvar = mvar_vec[m]
    mvar.inited || throw(ArgumentError("mvar is uninitialized: call init_nlvar!(...) first."))

    ψ = nlsol.ψ[m]
    b = nlsol.vtemp

    linapply!(b, mvar.lsd, ψ)  # b = A * ψ

    return norm(b)  # 2-norm
end

norm_nleq(m::Integer, nlsol::NonlasingSol, nlvar::NonlasingVar) = norm_nleq(m, nlsol, nlvar.mvar_vec)


# Unlike update_lsol!, this update_nlsol! takes the mode index m, because nonlasing mode
# equations are uncoupled between nonlasing modes.  This asymmetry between update_lsol! and
# update_nlsol! may need to be fixed because it makes tracking code difficult.
update_nlsol!(nlsol::NonlasingSol, m::Integer, nlvar::NonlasingVar) = update_nlsol!(nlsol, m, nlvar.mvar_vec[m])

function update_nlsol!(nlsol::NonlasingSol,
                       m::Integer,  # index of nonlasing mode of interest
                       mvar::NonlasingModalVar)  # must be already initialized
    mvar.inited || throw(ArgumentError("mvar is uninitialized: call init_nlvar!(...) first."))

    # Retrieve necessary variables for constructing the constraint.
    ψ = nlsol.ψ[m]
    iₐ = nlsol.iₐ[m]

    ∂f∂ω = mvar.∂f∂ω
    v = nlsol.vtemp
    linsolve!(v, mvar.lsd, ∂f∂ω)

    ∆ω = ψ[iₐ] / v[iₐ]
    nlsol.ω[m] += ∆ω
    ψ .= ∆ω .* v

    # @info "|∆ψ[iₐ]| = $(abs(ψ[iₐ] - 1))"
    @assert ψ[iₐ] ≈ 1
    ψ ./= ψ[iₐ]

    # Mark mvar uninitialized for the update solution.
    mvar.inited = false

    return nothing
end
