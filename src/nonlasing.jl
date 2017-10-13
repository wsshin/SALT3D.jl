export NonlasingSol, NonlasingVar
export init_nlvar!, norm_nleq, update_nlsol!

# Solutions to the nonlasing equation.
mutable struct NonlasingSol{VC<:AbsVecComplex}  # VC can be PETSc vector
    ω::VecComplex  # M complex numbers: frequencies of modes (M = # of nonlasing modes)
    ψ::Vector{VC}  # M complex vectors: normalized modes
    iₐ::VecInt  # M integers: row indices where amplitudes are measured
    function NonlasingSol{VC}(ω::AbsVecNumber,
                              ψ::AbsVec{<:AbsVecNumber},
                              iₐ::AbsVecInteger) where {VC<:AbsVecComplex}
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

        return new(ω, ψ, iₐ)
    end
end
NonlasingSol(ω::AbsVecNumber, ψ::AbsVec{VC}, iₐ::AbsVecInteger) where {VC<:AbsVecComplex} =
    NonlasingSol{VC}(ω, ψ, iₐ)
NonlasingSol(ω::AbsVecNumber, Ψ::AbsMatComplex, iₐ::AbsVecInteger) =
    (M = length(ω); NonlasingSol(ω, [Ψ[:,m] for m = 1:M], iₐ))

NonlasingSol(ω::AbsVecNumber, ψ::AbsVec{<:AbsVecComplex}) = NonlasingSol(ω, ψ, indmax.(map(x->abs.(x), ψ)))  # why no indmax(abs, array)?
NonlasingSol(ω::AbsVecNumber, Ψ::AbsMatComplex) = (M = length(ω); NonlasingSol(ω, [Ψ[:,m] for m = 1:M]))


# To do: check if the following works for vtemp of PETSc vector type.
NonlasingSol(vtemp::AbsVec,  # template vector with N entries
             M::Integer) =
    NonlasingSol(VecComlpex(M), [similar(vtemp,CFloat) for m = 1:M], VecInt(M))
NonlasingSol(N::Integer, M::Integer) = NonlasingSol(VecFloat(N), M)

Base.length(nlsol::NonlasingSol) = length(nlsol.ψ)
function Base.normalize!(nlsol::NonlasingSol)
    for m = 1:length(nlsol)
        ψ = nlsol.ψ[m]
        iₐ = nlsol.iₐ[m]
        ψ ./= ψ[iₐ]  # make ψ[iₐ] = 1
    end
end

# nonlasing reduced bar: D
# nonlasing modal var: γ, γ′, εeff, A

# Nonlasing equation variables that are unique to each nonlasing mode
mutable struct NonlasingModalVar{MC<:AbsMatComplex,VC<:AbsVecComplex}  # MC and VC can be PETSc matrix and vector
    A::MC
    ∂f∂ω::VC
    function NonlasingModalVar{MC,VC}(A::AbsMatNumber,  # dense A is automatically converted to sparse matrix if MC is sparse type
                                      ∂f∂ω::AbsVecNumber) where {MC<:AbsMatComplex,VC<:AbsVecComplex}
        N = length(∂f∂ω)
        size(A) == (N,N) || throw(ArgumentError("Each entry of size(A) = $(size(A)) and length∂f∂ω) = $N must be the same."))

        return new(A, ∂f∂ω)
    end
end
NonlasingModalVar(A::MC, ∂f∂ω::VC) where {MC<:AbsMatComplex,VC<:AbsVecComplex} =
    NonlasingModalVar{MC,VC}(A, ∂f∂ω)

# To do: check if the following works for vtemp of PETSc vector type.
NonlasingModalVar(mtemp::AbsMat,  # template N×N matrix (e.g., sparse matrix with all nonzero locations already specified)
                  vtemp::AbsVec) =  # template vector with N entries
    NonlasingModalVar(similar(mtemp,CFloat), similar(vtemp,CFloat))
NonlasingModalVar(mtemp::AbsMat) =
    (N = size(mtemp)[1]; NonlasingModalVar(similar(mtemp,CFloat), VecComplex(N), VecComplex(N)))


mutable struct NonlasingVar{MC<:AbsMatComplex,VC<:AbsVecComplex}
    mvar_vec::Vector{NonlasingModalVar{MC,VC}}
end
NonlasingVar(mtemp::AbsMat, vtemp::AbsVec, M::Integer) = NonlasingVar([NonlasingModalVar(mtemp, vtemp) for m = 1:M])
NonlasingVar(mtemp::AbsMat, M::Integer) = (N = size(mtemp)[1]; NonlasingVar(mtemp, VecFloat(N), M))


function init_modal_var!(mvar::NonlasingModalVar,
                         m::Integer,  # index of lasing mode
                         nlsol::NonlasingSol,
                         D::AbsVecFloat,  # population inversion
                         CC::AbsMatNumber,
                         param::SALTParam)
    ω = nlsol.ω[m]
    ψ = nlsol.ψ[m]

    γ = gain(ω, param.ωₐ, param.γ⟂)
    γ′ = gain′(ω, param.ωₐ, param.γ⟂)

    # Below, avoid allocations and use preallocated arrays in mvar.
    ε = mvar.∂f∂ω  # temporary storage for effective permitivity: εc + γ D
    ε .= param.εc .+ γ .* D

    # Move A, rowA⁻¹ᵢₐ away.  These need to be used only
    create_A!(mvar.A, CC, ω, ε)
    mvar.∂f∂ω .= (2ω .* param.εc + (2ω*γ + ω^2*γ′) .* D) .* ψ  # derivative of nonlasing equation function w.r.t ω

    return nothing
end

init_nlvar!(nlvar::NonlasingVar, m::Integer, nlsol::NonlasingSol, D::AbsVecFloat, CC::AbsMatNumber, param::SALTParam) =
    init_modal_var!(nlvar.mvar_vec[m], m, nlsol, D, CC, param)

function norm_nleq(nlsol::NonlasingSol, m::Integer, mvar::NonlasingModalVar)
    A = mvar.A
    ψ = nlsol.ψ[m]
    return norm(A*ψ)  # 2-norm
end
norm_nleq(nlsol::NonlasingSol, m::Integer, nlvar::NonlasingVar) = norm_nleq(nlsol, m, nlvar.mvar_vec[m])


function update_nlsol!(nlsol::NonlasingSol,
                       m::Integer,  # index of nonlasing mode of interest
                       mvar::NonlasingModalVar)  # must be already initialized
    # Retrieve necessary variables for constructing the constraint.
    ψ = nlsol.ψ[m]
    iₐ = nlsol.iₐ[m]
    ψᵢₐold = ψ[iₐ]

    ∂f∂ω = mvar.∂f∂ω
    v = mvar.A \ ∂f∂ω

    ∆ω = ψ[iₐ] / v[iₐ]
    nlsol.ω[m] += ∆ω
    ψ .= ∆ω .* v

    # info("|∆ψ[iₐ]| = $(abs(ψ[iₐ] - ψᵢₐold))")
    assert(ψ[iₐ] ≈ ψᵢₐold)
    ψ ./= ψ[iₐ]

    return nothing
end
update_nlsol!(nlsol::NonlasingSol, m::Integer, nlvar::NonlasingVar) = update_nlsol!(nlsol, m, nlvar.mvar_vec[m])
