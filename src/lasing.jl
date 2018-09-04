# Define functions to find ∆ω, ∆a, ∆ψ by the implicit Newton step calculation algorithm.
# Note that ω, a, ψ are fixed in this calculation.

# We can save computations by implementing the assumption ∆D = 0, and also by updating ψ
# directly without calculating ∆ψ.

export LasingSol, LasingVar

# Below (and also in nonlasing.jl), many constructors take some arguments as templates, and
# those template arguments are not directly set as fields: separate copies are created
# (usually by `similar`) and set as fields.
#
# - An alternative would be to take parameters for creating those fields and create the
# fields.  E.g., we could take the length and the element type of a vector, create a vector,
# and set it as a field.  However, we are aiming to support many different types of vectors
# (e.g., PETSc vector).  We cannot predict all the parameters needed for creating these
# vectors.  By taking an actual instance of a vector in this case, we can support any types
# of vectors.
#
# - Then, the next possibility would be to let the users to provide a copy, instead of
# automatically generating a copy inside code.  I avoid taking this route for consistency in
# regard to automatic type conversion inside constructors.  Suppose I create a custom type
# named `MyType` containing `v::Vector{Float64}` as a field.  If I instantiate `MyType` by
# giving a `vi::Vector{Int64}` as an argument, `vi` is automatically type-converted into a
# Vector{Float64} and set to `v`.  Here, the resulting `v::Vector{Float64}` is a separate
# copy from `vi::Vector{Int64}`.  On the other hand, if I instantiate `MyType` by giving a
# `vf::Vector{Float64}` as an argument, a copy is not created and the `vf` itself is set to
# `v`.  This inconsistency is confusing, because in one case changing the contents of the
# outside vector `vi` does not affect the `MyType` instance, whereas in the other case it
# does.  In order to avoid this inconsistency, I decide to always create a copy of a
# template argument.  This way, no matter whether automatic type conversion occurs or not,
# the stored field is a separate copy from the argument given to a constructor.


# Solutions to the lasing equation.
mutable struct LasingSol{VC<:AbsVecComplex}  # VC can be PETSc vector
    ω::VecFloat  # M real numbers: frequencies of modes (M = # of lasing modes)
    a²::VecFloat  # M real numbers: squared "amplitudes" of modes inside hole-burning term; this includes γ(ω)² factor
    ψ::Vector{VC}  # M complex vectors: normalized modes
    iₐ::VecInt  # M integers: row indices where amplitudes are measured
    vtemp::VC  # temporary storage for N complex numbers; note its contents can change at any point
    activated::VecBool  # activated[m] == true if mode m is just activated; used in simulate!
    active::VecBool  # active[m] == true if mode m is active (i.e., lasing)
    m_active::VecInt  # vector of active (i.e., lasing) mode indices; collection of m such that active[m] == true
    function LasingSol{VC}(ω::AbsVecReal,
                           a²::AbsVecReal,
                           ψ::AbsVec{<:AbsVecNumber},
                           iₐ::AbsVecInteger,
                           vtemp::AbsVecNumber) where {VC<:AbsVecComplex}
        length(ω)==length(a²)==length(ψ)==length(iₐ) ||
            throw(ArgumentError("length(ω) = $(length(ω)), length(a²) = $(length(a²)),
                                 length(ψ) = $(length(ψ)), length(iₐ) = $(length(iₐ)) must be the same."))

        M = length(ψ)
        if M > 0
            N = length(ψ[1])
            for m = 2:M
                length(ψ[m]) == N ||
                    throw(ArgumentError("length(ψ[$m]) = $(length(ψ[m])) and length(ψ[1]) = $N must be the same"))
            end
        end

        return new(ω, a², ψ, iₐ, vtemp, fill(false,M), fill(false,M), VecInt(0))  # active[m]=false for all m: no mode is lasing
    end
end
LasingSol(ω::AbsVecReal, a²::AbsVecReal, ψ::AbsVec{VC}, iₐ::AbsVecInteger, vtemp::VC) where {VC<:AbsVecComplex} =
    LasingSol{VC}(ω, a², ψ, iₐ, vtemp)

# To do: check if the following works for vtemp of PETSc vector type.
LasingSol(vtemp::AbsVec, M::Integer) =  # vtemp has N entries
    LasingSol(zeros(M), zeros(M), [similar(vtemp,CFloat).=0 for m = 1:M], zeros(Int,M), similar(vtemp,CFloat))
LasingSol(N::Integer, M::Integer) = LasingSol(VecFloat(N), M)

Base.length(lsol::LasingSol) = length(lsol.ψ)

function Base.normalize!(lsol::LasingSol)
    for m = lsol.m_active
        ψ = lsol.ψ[m]
        iₐ = lsol.iₐ[m]
        ψ ./= ψ[iₐ]  # make ψ[iₐ] = 1
    end
end


# ∆'s of the solutions to the lasing equation
mutable struct ∆LasingSol{VC<:AbsVecComplex}  # VC can be PETSc vector
    ∆ω::VecFloat  # M real numbers (M = # of lasing modes)
    ∆a²::VecFloat  # M real numbers
    ∆ψ::Vector{VC}  # M complex vectors (each with N complex numbers)
    vtemp::VC  # temporary storage for N complex numbers; note its contents can change at any point
    function ∆LasingSol{VC}(∆ω::AbsVecReal, ∆a²::AbsVecReal, ∆ψ::AbsVec{<:AbsVecNumber}, vtemp::AbsVecNumber) where {VC<:AbsVecComplex}
        length(∆ω)==length(∆a²)==length(∆ψ) ||
            throw(ArgumentError("length(∆ω) = $(length(∆ω)), length(∆a²) = $(length(∆a²)),
                                 length(∆ψ) = $(length(∆ψ)) must be the same."))
        M = length(∆ψ)
        N = length(vtemp)
        for m = 1:M
            length(∆ψ[m]) == N ||
                throw(ArgumentError("length(∆ψ[$m]) = $(length(∆ψ[m])) and length(vtemp) = $N must be the same."))
        end

        return new(∆ω, ∆a², ∆ψ, vtemp)
    end
end
∆LasingSol(∆ω::AbsVecReal, ∆a²::AbsVecReal, ∆ψ::AbsVec{VC}, vtemp::VC) where {VC<:AbsVecComplex} =
    ∆LasingSol{VC}(∆ω, ∆a², ∆ψ, vtemp)

# To do: check if the following works for vtemp of PETSc vector type.
∆LasingSol(vtemp::AbsVec, M::Integer) =  # vtemp has N entries
    ∆LasingSol(VecFloat(M), VecFloat(M), [similar(vtemp,CFloat) for m = 1:M], similar(vtemp,CFloat))
∆LasingSol(N::Integer, M::Integer) = ∆LasingSol(VecFloat(N), M)

# Lasing equation variables that are reduced from all lasing modes
mutable struct LasingReducedVar{VF<:AbsVecFloat}  # VF can be PETSc vector
    D::VF
    D′::VF
    ∆D::VF
    ∇ₐ₂D::Vector{VF}
    function LasingReducedVar{VF}(D::AbsVecReal, D′::AbsVecReal, ∆D::AbsVecReal, ∇ₐ₂D::AbsVec{<:AbsVecReal}) where {VF<:AbsVecFloat}
        length(D)==length(D′)==length(∆D) ||
            throw(ArgumentError("length(D) = $(length(D)), length(D′) = $(length(D′)),
                                 length(∆D) = $(length(∆D)) must be the same."))
        M = length(∇ₐ₂D)
        N = length(D)
        for m = 1:M
            length(∇ₐ₂D[m]) == N ||
                throw(ArgumentError("length(∇ₐ₂D[$m]) = $(length(∇ₐ₂D[m])) and length(D) = $N must be the same."))
        end

        return new(D, D′, ∆D, ∇ₐ₂D)
    end
    # function LasingReducedVar{VF}(D::AbsVecReal, D′::AbsVecReal, ∇ₐ₂D::AbsVec{<:AbsVecReal}) where {VF<:AbsVecFloat}
    #     length(D)==length(D′) ||
    #         throw(ArgumentError("length(D) = $(length(D)) and length(D′) = $(length(D′)) must be the same."))
    #     N = length(D)
    #     M = length(∇ₐ₂D)
    #     for m = 1:M
    #         length(∇ₐ₂D[m]) == N ||
    #             throw(ArgumentError("length(∇ₐ₂D[$m]) = $(length(∇ₐ₂D[m])) and length(D) = $N must be the same."))
    #     end
    #
    #     return new(D, D′, ∇ₐ₂D)
    # end
end
LasingReducedVar(D::AbsVecReal, D′::AbsVecReal, ∆D::AbsVecReal, ∇ₐ₂D::AbsVec{VF}) where {VF<:AbsVecFloat} =
    LasingReducedVar{VF}(D, D′, ∆D, ∇ₐ₂D)
# LasingReducedVar(D::AbsVecReal, D′::AbsVecReal, ∇ₐ₂D::AbsVec{VF}) where {VF<:AbsVecReal} =
#     LasingReducedVar{VF}(D, D′, ∇ₐ₂D)

# To do: check if the following works for vtemp of PETSc vector type.
LasingReducedVar(vtemp::AbsVec, M::Integer) =  # vtemp has N entries
    LasingReducedVar(similar(vtemp,Float), similar(vtemp,Float), similar(vtemp,Float), [similar(vtemp,Float) for m = 1:M])
LasingReducedVar(N::Integer, M::Integer) =  LasingReducedVar(VecFloat(N), M)


# Lasing equation variables that are unique to each lasing mode
mutable struct LasingModalVar{LSD<:LinearSolverData,VC<:AbsVecComplex}  # VC can be PETSc vector
    # Consider adding ε as a field, in case the user wants to solve the "linear" eigenvalue
    # equation for some reason.
    lsd::LSD
    ω²γψ::VC
    ∂f∂ω::VC
    function LasingModalVar{LSD,VC}(lsd::LSD,
                                   ω²γψ::AbsVecNumber,
                                   ∂f∂ω::AbsVecNumber) where {LSD<:LinearSolverData,VC<:AbsVecComplex}
        length(ω²γψ)==length(∂f∂ω) ||
            throw(ArgumentError("length(ω²γψ) = $(length(ω²γψ)) and length(∂f∂ω) = $(length(∂f∂ω)) must be the same."))
        N = length(ω²γψ)

        return new(lsd, ω²γψ, ∂f∂ω)
    end
end
LasingModalVar(lsd::LSD, ω²γψ::VC, ∂f∂ω::VC) where {LSD<:LinearSolverData,VC<:AbsVecComplex} =
    LasingModalVar{LSD,VC}(lsd, ω²γψ, ∂f∂ω)

# To do: check if the following works for vtemp of PETSc vector type.
LasingModalVar(lsd_temp::LinearSolverData, vtemp::AbsVec) =  # vtemp has N entries
    LasingModalVar(similar(lsd_temp), similar(vtemp,CFloat), similar(vtemp,CFloat))
LasingModalVar(lsd_temp::LinearSolverData, N::Integer) =
    LasingModalVar(similar(lsd_temp), VecComplex(N), VecComplex(N))

mutable struct LasingConstraint
    A::MatFloat  # constraint matrix
    b::VecFloat  # constraint vector
    m2_active::VecBool  # vector of 2M booleans; true if corresponding row and column are active
    LasingConstraint(M::Integer) = new(MatFloat(2M,2M), VecFloat(2M), VecBool(2M))
end

function activate!(cst::LasingConstraint, lsol::LasingSol)
    M = length(lsol)
    for m = 1:M
        cst.m2_active[2m-1] = cst.m2_active[2m] = lsol.active[m]
    end
end


# Calculate the change induced in population inversion by the change in ψ's.
# Note that this is the only function in this file whose output depends on ∆ψ's.
function ∆popinv!(∆D::AbsVecFloat,  # output
                  D′::AbsVecReal,  # derivative of population inversion; output of popinv′
                  ∆lsol::∆LasingSol,
                  lsol::LasingSol)
    ∆D .= 0  # initialize
    for m = lsol.m_active
        ∆D .+= 2lsol.a²[m] .* real.(conj.(lsol.ψ[m]) .* ∆lsol.∆ψ[m])
    end
    ∆D .*= D′

    return nothing
end


# Calculate the gradient of population inversion with respect to amplitudes a.
function ∇ₐ₂popinv!(∇ₐ₂D::AbsVec{<:AbsVecFloat},  # output
                    D′::AbsVecReal,  # derivative of population inversion; output of popinv′
                    lsol::LasingSol)
    for m = lsol.m_active
        ∇ₐ₂D[m] .= abs2.(lsol.ψ[m]) .* D′
    end

    return nothing
end


# Components necessary for fixed-point iteration
mutable struct LasingVar{LSD<:LinearSolverData,VC<:AbsVecComplex,VF<:AbsVecFloat}
    ∆lsol::∆LasingSol{VC}
    mvar_vec::Vector{LasingModalVar{LSD,VC}}
    rvar::LasingReducedVar{VF}
    cst::LasingConstraint
    inited::Bool
end
LasingVar(lsd_temp::LinearSolverData, vtemp::AbsVec, M::Integer) =
    LasingVar(∆LasingSol(vtemp, M), [LasingModalVar(lsd_temp, vtemp) for m = 1:M], LasingReducedVar(vtemp, M), LasingConstraint(M), false)
LasingVar(lsd_temp::LinearSolverData, N::Integer, M::Integer) = LasingVar(lsd_temp, VecFloat(N), M)

function init_reduced_var!(rvar::LasingReducedVar, ∆lsol::∆LasingSol, lsol::LasingSol, param::SALTParam)
    hb = lsol.vtemp  # temporary storage for hole-burning term
    hole_burning!(hb, lsol.a², lsol.ψ)

    rvar.D .= param.D₀ ./ hb  # D = D₀ / (1 + ∑a²|ψ|²)
    rvar.D′ .= -param.D₀ ./ abs2.(hb)  # D′(∑a²|ψ|²) = -D₀ / (1+∑a²|ψ|²)²

    ∆popinv!(rvar.∆D, rvar.D′, ∆lsol, lsol)  # comment this out when ∆ψ_old = 0 (so that ∆D = 0)
    ∇ₐ₂popinv!(rvar.∇ₐ₂D, rvar.D′, lsol)

    # info("‖D‖ = $(norm(rvar.D)), ‖D′‖ = $(norm(rvar.D′)), ‖∆D‖ = $(norm(rvar.∆D))")
    # for m = lsol.m_active
    #     info("‖∇ₐ₂D[$m]‖ = $(norm(rvar.∇ₐ₂D[m]))")
    # end

    return nothing
end


function init_modal_var!(mvar::LasingModalVar,
                         m::Integer,  # index of lasing mode
                         lsol::LasingSol,
                         rvar::LasingReducedVar,
                         param::SALTParam)
    isreal(lsol.ω[m]) || throw(ArgumentError("lsol.ω[$m] = $(lsol.ω[m]) must be real."))
    ω = real(lsol.ω[m])

    γ = gain(ω, param.ωₐ, param.γ⟂)
    γ′ = gain′(ω, param.ωₐ, param.γ⟂)

    ε = lsol.vtemp  # temporary storage for effective permitivity: εc + γ D
    ε .= param.εc .+ γ .* rvar.D

    init_lsd!(mvar.lsd, ω, ε)
    mvar.ω²γψ .= (ω^2*γ) .* lsol.ψ[m]
    mvar.∂f∂ω .= (2ω .* ε + (ω^2*γ′) .* rvar.D) .* lsol.ψ[m]  # derivative of lasing equation function w.r.t ω

    return nothing
end


# Initialize ∆lsol by filling zeros to all vectors for the single-loop algorithm.
function init_∆lsol!(∆lsol::∆LasingSol)
    ∆lsol.∆ω .= 0
    ∆lsol.∆a² .= 0
    for ∆ψₘ = ∆lsol.∆ψ
        ∆ψₘ .= 0
    end

    return nothing
 end


function init_lvar!(lvar::LasingVar, lsol::LasingSol, param::SALTParam)
    ∆lsol = lvar.∆lsol
    mvar_vec = lvar.mvar_vec
    rvar = lvar.rvar

    init_∆lsol!(∆lsol)  # make ∆lsol all zero
    init_reduced_var!(rvar, ∆lsol, lsol, param)
    for m = lsol.m_active
        init_modal_var!(mvar_vec[m], m, lsol, rvar, param)
    end

    lvar.inited = true

    return nothing
end


function norm_leq_impl(lsol::LasingSol, mvar_vec::AbsVec{<:LasingModalVar})
    leq² = 0.0
    b = lsol.vtemp
    for m = lsol.m_active
        lsd = mvar_vec[m].lsd
        ψ = lsol.ψ[m]
        linapply!(b, lsd, ψ)  # b = A * ψ
        leq² = max(leq², sum(abs2,b))  # 2-norm for each mode, 1-norm between modes
    end

    return √leq²  # return 0.0 if lsol.m_active is empty
end

function norm_leq(lsol::LasingSol, lvar::LasingVar, param::SALTParam)
    lvar.inited || throw(ArgumentError("lvar is uninitialized: call init_lvar!(...) first."))

    return norm_leq_impl(lsol, lvar.mvar_vec)
end

# Create the nth constraint equation on ∆ω and ∆a.
function set_constraint!(cst::LasingConstraint,
                         ∆lsol::∆LasingSol,
                         lsol::LasingSol,
                         m::Integer,  # index of lasing mode of interest
                         mvar::LasingModalVar,  # modal variables for mth lasing mode
                         rvar::LasingReducedVar)
    ψ = lsol.ψ[m]
    ∆ψ = ∆lsol.∆ψ[m]
    iₐ = lsol.iₐ[m]

    vtemp1 = lsol.vtemp
    vtemp2 = ∆lsol.vtemp

    N = length(ψ)

    # Retrieve necessary variables for constructing the constraint.
    ∆D = rvar.∆D
    ∇ₐ₂D = rvar.∇ₐ₂D

    # Calculate the iₐth row of A⁻¹ and keep it as a column vector
    eᵢₐ = vtemp2
    eᵢₐ .= 0
    eᵢₐ[iₐ] = 1
    r = vtemp1  # storage for row vector
    linsolve_transpose!(r, mvar.lsd, eᵢₐ)

    ω²γψ = mvar.ω²γψ
    ∂f∂ω = mvar.∂f∂ω
    A = cst.A  # constraint right-hand-side matrix
    b = cst.b  # constraint right-hand-side vector

    # Set the right-hand-side vector of the constraint.
    vtemp2 .= ∆D.*ω²γψ  # must be 0 for single-loop algorithm
    ζv = ψ[iₐ] - BLAS.dotu(r, vtemp2)  # scalar; note negation because ζv is quantity on RHS
    # info("ψ[iₐ] = $(ψ[iₐ]), ‖ψ‖ = $(norm(ψ))")
    # ζv = ψ[iₐ]  # ∆D = 0
    b[2m-1] = real(ζv)
    b[2m] = imag(ζv)

    # Set the left-hand-side matrix of the constraint.
    ζω = BLAS.dotu(r, ∂f∂ω)
    A[2m-1,2m-1] = real(ζω)
    A[2m,2m-1] = imag(ζω)

    for j = lsol.m_active
        vtemp2 .= ∇ₐ₂D[j] .* ω²γψ  # this uses no allocations
        ζa² = BLAS.dotu(r, vtemp2)
        A[2m-1,2j] = real(ζa²)
        A[2m,2j] = imag(ζa²)
    end

    # info("A = $A, b = $b")

    return nothing
end


# Move lsolₘ by ∆lsolₘ.
function apply_∆solₘ!(lsol::LasingSol,
                      ∆lsol::∆LasingSol,
                      m::Integer,  # index of lasing mode of interest
                      mvar::LasingModalVar,  # ∆ω and ∆a² must be already updated; see update_∆lsol
                      rvar::LasingReducedVar)
    # Retrieve necessary variables for constructing the constraint.
    ∆D = rvar.∆D
    ∇ₐ₂D = rvar.∇ₐ₂D

    ω²γψ = mvar.ω²γψ
    ∂f∂ω = mvar.∂f∂ω

    ∆ω = ∆lsol.∆ω[m]
    ∆a² = ∆lsol.∆a²
    ∆ψ = ∆lsol.∆ψ[m]
    # info("‖∆D‖ = $(norm(∆D)), ‖ω²γψ‖ = $(norm(ω²γψ)), ‖∂f∂ω‖ = $(norm(∂f∂ω)), ∆ω = $∆ω, ∆a² = $∆a², ‖∆ψ‖ = $(norm(∆ψ))")

    ψ = lsol.ψ[m]

    # Calculate the vector to feed to A⁻¹.
    vtemp = lsol.vtemp
    vtemp .= (∆D .* ω²γψ) .+ (∆ω .* ∂f∂ω)
    # vtemp .= (∆ω .* ∂f∂ω)  # because ∆D = 0
    for j = lsol.m_active
        # info("‖∇ₐ₂D[$j]‖ = $(norm(∇ₐ₂D[j]))")
        vtemp .+= ∆a²[j] .* (∇ₐ₂D[j] .* ω²γψ)
    end

    # # Calculate ∆ψ.
    # ∆ψ .= mvar.A \ vtemp
    # ∆ψ .-= ψ
    # # ∆ψ[lsol.iₐ[m]] = 0

    # info("‖A‖₁ = $(norm(mvar.A,1)), ‖vtemp‖ = $(norm(vtemp))")
    linsolve!(ψ, mvar.lsd, vtemp)
    # ψ .= mvar.A \ vtemp

    # Normalized ψ just in case ψ[iₐ] ≠ 1.  (Do we need to unnormalize a²?)
    iₐ = lsol.iₐ[m]
    ψ ./= ψ[iₐ]

    # The following could have been updated before this function, because they were already
    # prepared.
    lsol.ω[m] += ∆ω
    lsol.a²[m] += ∆a²[m]

    return nothing
end


# lvar must be already initialized by init_lvar! before starting the fixed-point iteration.
function update_lsol!(lsol::LasingSol, lvar::LasingVar, param::SALTParam)
    lvar.inited || throw(ArgumentError("lvar is uninitialized: call init_lvar!(...) first."))
    update_lsol_impl!(lsol, lvar.∆lsol, lvar.mvar_vec, lvar.rvar, lvar.cst, param)
    lvar.inited = false

    return nothing
end


# Fixed-point equation for lsol.  Calculate ∆ω and ∆a, and then ∆ψ, and move lsol by them.
#
# This function is expensive to call, because it involves 2Mₗ linear solves, where Mₗ is the
# number of lasing modes.  Therefore, try not to call this function unnecessarily.
#
# init_lvar! must be called before using this function to make lvar prepared.  However,
# init_lvar! is not exported in order to force checking the norm by norm_leq to avoid
# calling this function when the norm is small enough.
function update_lsol_impl!(lsol::LasingSol,
                           ∆lsol::∆LasingSol,
                           mvar_vec::AbsVec{<:LasingModalVar},  # must be already initialized
                           rvar::LasingReducedVar,  # must be already initialized
                           cst::LasingConstraint,
                           param::SALTParam)
    # Construct the constraint equation on ∆ω and ∆a.
    cst.A .= 0
    cst.b .= 0
    for m = lsol.m_active
        set_constraint!(cst, ∆lsol, lsol, m, mvar_vec[m], rvar)
    end

    # Calculate ∆ω and ∆a.
    activate!(cst, lsol)
    ind = cst.m2_active
    ∆ωa² = cst.A[ind,ind] \ cst.b[ind]
    # info("cst.A = $(cst.A[ind,ind]), cst.b = $(cst.b[ind]), ∆ωa² = $∆ωa²")
    c = 0  # count
    for m = lsol.m_active
        c += 1
        ∆lsol.∆ω[m] = ∆ωa²[2c-1]
        ∆lsol.∆a²[m] = ∆ωa²[2c]
    end

    # Update ∆ψ.
    # info("lsol.ω = $(lsol.ω), lsol.a² = $(lsol.a²), lsol.m_active = $(lsol.m_active)")
    for m = lsol.m_active
        # info("before apply: ‖lsol.ψ[$m]‖ = $(norm(lsol.ψ[m]))")
        apply_∆solₘ!(lsol, ∆lsol, m, mvar_vec[m], rvar)
        # info("after apply: ‖lsol.ψ[$m]‖ = $(norm(lsol.ψ[m]))")
    end

    return nothing
end


# To use andersonaccel!, implement anderson_SALT! that accepts SALTSol as an initial guess
# and g! that takes SALTSol and returns SALTSol.  anderson_SALT! must create a version of g!
# that takes and returns vectors using CatViews.

# I will need to implement `reinterpret` for PETSc vectors to view complex PETSc vectors as
# a real PETSc vector.

function lsol2rvec(lsol::LasingSol)
    m_active = lsol.m_active
    ψr = reinterpret.(Float, lsol.ψ[lsol.m_active])
    # ψr = lsol.ψ[lsol.m_active]  # complex version

    return CatView(lsol.ω[lsol.m_active], lsol.a²[lsol.m_active], ψr...)
    # return CatView(ψr...)
    # return CatView(lsol.ω[lsol.m_active], lsol.a²[lsol.m_active])
end
