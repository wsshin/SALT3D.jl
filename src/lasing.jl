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
mutable struct LasingSol{VF<:AbsVecFloat,VC<:AbsVecComplex}  # VF and VC can be PETSc vectors with N entries
    ω::VecFloat  # M real numbers: frequencies of modes (M = # of total modes, including both lasing and nonlasing)
    a²::VecFloat  # M real numbers: squared "amplitudes" of modes inside hole-burning term; this includes γ(ω)² factor
    ψ::Vector{VC}  # M complex vectors: normalized modes
    abs2ψ::Vector{VF}  # M complex vectors: absolute squares of normalized modes
    iₐ::VecInt  # M integers: row indices where amplitudes are measured
    vtemp::VC  # temporary storage for N complex numbers
    wtemp::VC  # temporary storage for N complex numbers
    activated::VecBool  # activated[m] == true if mode m is just activated; used in simulate!
    active::VecBool  # active[m] == true if mode m is active (i.e., lasing)
    m_active::VecInt  # vector of active (i.e., lasing) mode indices; collection of m such that active[m] == true
    function LasingSol{VF,VC}(ω::AbsVecReal,
                              a²::AbsVecReal,
                              ψ::AbsVec{<:AbsVecNumber},
                              abs2ψ::AbsVec{<:AbsVecReal},
                              iₐ::AbsVecInteger,
                              vtemp::AbsVecComplex) where {VF<:AbsVecFloat,VC<:AbsVecComplex}
        M = length(ω)
        length(a²)==length(ψ)==length(abs2ψ)==length(iₐ)==M ||
            throw(ArgumentError("length(ω) = $M, length(a²) = $(length(a²)), length(ψ) = $(length(ψ))" *
                                 "length(abs2ψ) = $(length(abs2ψ)), length(iₐ) = $(length(iₐ)) must be the same."))

        N = length(vtemp)
        for m = 1:M
            length(ψ[m])==length(abs2ψ[m])==N ||
                throw(ArgumentError("length(ψ[$m]) = $(length(ψ[m])), length(abs2ψ[$m]) = $(length(abs2ψ[m])), length(vtemp) = $N must be the same."))
        end

        return new(ω, a², ψ, abs2ψ, iₐ, vtemp, copy(vtemp), fill(false,M), fill(false,M), VecInt(undef,0))  # active[m]=false for all m: no mode is lasing
    end
end
LasingSol(ω::AbsVecReal, a²::AbsVecReal, ψ::AbsVec{VC}, abs2ψ::AbsVec{VF}, iₐ::AbsVecInteger, vtemp::VC) where {VF<:AbsVecFloat,VC<:AbsVecComplex} =
    LasingSol{VF,VC}(ω, a², ψ, abs2ψ, iₐ, vtemp)

# To do: check if the following works for vtemp of PETSc vector type.
LasingSol(vtemp::AbsVec, M::Integer) =  # vtemp has N entries
    LasingSol(zeros(M), zeros(M), [similar(vtemp,CFloat).=0 for m = 1:M], [similar(vtemp,Float).=0 for m = 1:M], zeros(Int,M), similar(vtemp,CFloat))
LasingSol(N::Integer, M::Integer) = LasingSol(VecFloat(undef,N), M)

Base.length(lsol::LasingSol) = length(lsol.ω)  # number of total modes, including both lasing and nonlasing
num_active(lsol::LasingSol) = sum(lsol.active)  # number of modes currently lasing


# Note that this function changes iₐ.  Therefore, this must not be called inside the
# iteration for finding the solution for a given pump strength, because iₐ must be kept the
# same for a given pump strength in order to solve the equations with the same normalization
# conditions.
function LinearAlgebra.normalize!(lsol::LasingSol)
    for m = lsol.m_active
        ψ = lsol.ψ[m]
        iₐ = argmax(abs2, ψ)
        lsol.iₐ[m] = iₐ

        lsol.a²[m] *= abs2(ψ[iₐ])
        ψ ./= ψ[iₐ]  # make ψ[iₐ] = 1
    end
end


# Lasing equation variables that are reduced from all lasing modes
mutable struct LasingReducedVar{VF<:AbsVecFloat}  # VF can be PETSc vector with N entries
    D::Vector{VF}  # length-K vector: each entry is N-vector storing population inversion, where K is number of atomic classes
    D′::Vector{VF}  # length-K vector: each entry is N-vector storing derivative of population inversion with respect to hole-burning term
    function LasingReducedVar{VF}(D::AbsVec{<:AbsVecReal},
                                  D′::AbsVec{<:AbsVecReal}) where {VF<:AbsVecFloat}
        K = length(D)  # number of atomic classes
        length(D′)==K ||
            throw(ArgumentError("length(D) = $K and length(D′) = $(length(D′)) must be the same"))

        if K ≥ 1
            N = length(D[1])
            for k = 1:K
                length(D[k])==length(D′[k])==N ||
                    throw(ArgumentError("length(D[$k]) = $(length(D[k])), length(D′[$k]) = $(length(D′[k])) must be $N."))
            end
        end

        return new(D, D′)
    end
end
LasingReducedVar(D::AbsVec{<:AbsVecReal}, D′::AbsVec{<:AbsVecReal}) = LasingReducedVar{eltype(D)}(D, D′)

# To do: check if the following works for vtemp of PETSc vector type.
LasingReducedVar(vtemp::AbsVec, K::Integer) = LasingReducedVar([similar(vtemp,Float).=0 for k=1:K], [similar(vtemp,Float).=0 for k=1:K])  # vtemp has N entries
LasingReducedVar(N::Integer, K::Integer) =  LasingReducedVar(VecFloat(undef,N), K)


# Initialize the variables that are NOT specific to a specific lasing mode.  These variables
# are constructed by summing up the contributions from all lasing modes.
function init_reduced_var!(rvar::LasingReducedVar, lsol::LasingSol, gp::GainProfile)
    # Update abs2ψ.
    for m = lsol.m_active
        lsol.abs2ψ[m] .= abs2.(lsol.ψ[m])
    end

    hb = lsol.vtemp  # temporary storage for hole-burning term
    for k = 1:length(gp)  # atomic classes
        # Calculate the (1 + hole-burning term) for atomic class k.
        abs2γ = gp.abs2gain[k]  # |γ²| function for atomic class k
        hole_burning!(hb, abs2γ, lsol.ω, lsol.a², lsol.abs2ψ)

        # Calculate the population inversion and the derivative of the population inversion
        # for the class-k atoms.
        #
        # hb is a complex vector storing real, and it is faster to take its real part first
        # and perform the further calculation on it.
        wt = gp.wt[k]  # weight function for atomic class k
        rvar.D[k] .= wt.(gp.D₀) ./ real.(hb)
        rvar.D′[k] .= (-1.0) .* wt.(gp.D₀) ./ abs2.(real.(hb))
    end

    return nothing
end


# Lasing equation variables that are unique to each lasing mode
mutable struct LasingModalVar{LSD<:LinearSolverData,VC<:AbsVecComplex}  # VC can be PETSc vector
    # Consider adding εeff as a field, in case the user wants to solve the "linear" eigenvalue
    # equation for some reason.
    lsd::LSD
    ∂f∂ω::VC
end

# To do: check if the following works for vtemp of PETSc vector type.
LasingModalVar(lsd_temp::LinearSolverData, vtemp::AbsVec) =  # for, e.g., PETSc vector vtemp; vtemp has N entries
    LasingModalVar(similar(lsd_temp), similar(vtemp,CFloat))

LasingModalVar(lsd_temp::LinearSolverData, N::Integer) =
    LasingModalVar(similar(lsd_temp), VecComplex(undef,N))


# Initialize the variables that are specific to a specific lasing mode.
function init_modal_var!(mvar::LasingModalVar,
                         m::Integer,  # index of lasing mode
                         lsol::LasingSol,
                         D::AbsVec{<:AbsVecFloat},
                         gp::GainProfile,
                         εc::AbsVecComplex)
    init_modal_var_impl!(mvar.lsd, mvar.∂f∂ω, lsol.vtemp, lsol.ω[m], lsol.ψ[m], D, gp, εc)
end

# This is used in nonlasing.jl as well.
function init_modal_var_impl!(lsd::LinearSolverData,
                              ∂f∂ω::AbsVecComplex,
                              vtemp::AbsVecComplex,
                              ω::Number,
                              ψ::AbsVecComplex,
                              D::AbsVec{<:AbsVecFloat},
                              gp::GainProfile,
                              εc::AbsVecComplex)
    K = length(gp)  # number of atomic classes
    γ = gp.gain  # K-vector whose kth entry is γ function of atomic class k
    γ′ = gp.gain′  # K-vector whose kth entry is γ′ function of atomic class k

    # Calculate the effective permittivity.
    εeff = vtemp  # temporary storage for effective permitivity: εc + ∑ₖ(γₖ Dₖ)
    εeff .= εc
    for k = 1:K
        εeff .+= γ[k](ω) .* D[k]
    end

    init_lsd!(lsd, ω, εeff)

    # Calculate ∂f∂ω: derivative of SALT function w.r.t. ω, ignoring ω-dependence of D.
    ∂f∂ω .= 2ω .* εeff
    for k = 1:K
        σ = ω^2 * γ′[k](ω)  # scalar
        ∂f∂ω .+= σ .* D[k]
    end
    ∂f∂ω .*= ψ

    return nothing
end


mutable struct LasingConstraint
    A::MatFloat  # constraint matrix
    b::VecFloat  # constraint vector
    ∆ωa²::VecFloat  # vector of calculated ∆ω and ∆a²: ∆ω[m] = ∆ωa²[2m-1] and ∆a²[m] = ∆ωa²[2m]
    LasingConstraint(M::Integer) = new(MatFloat(undef,2M,2M), VecFloat(undef,2M), VecFloat(undef,2M))
end


# Components necessary for fixed-point iteration
mutable struct LasingVar{LSD<:LinearSolverData,VF<:AbsVecFloat,VC<:AbsVecComplex}
    rvar::LasingReducedVar{VF}
    mvar_vec::Vector{LasingModalVar{LSD,VC}}
    cst::LasingConstraint
    inited::Bool
end
LasingVar(lsd_temp::LinearSolverData, vtemp::AbsVec, K::Integer, M::Integer) =  # K: number of atomic classes, M: number of modes
    LasingVar(LasingReducedVar(vtemp, K), [LasingModalVar(lsd_temp, vtemp) for m = 1:M], LasingConstraint(M), false)
LasingVar(lsd_temp::LinearSolverData, N::Integer, K::Integer, M::Integer) = LasingVar(lsd_temp, VecFloat(undef,N), K, M)


function init_lvar!(lvar::LasingVar, lsol::LasingSol, gp::GainProfile, εc::AbsVecComplex)
    rvar = lvar.rvar
    mvar_vec = lvar.mvar_vec

    init_reduced_var!(rvar, lsol, gp)
    for m = lsol.m_active
        init_modal_var!(mvar_vec[m], m, lsol, rvar.D, gp, εc)
    end

    lvar.inited = true

    return nothing
end


function norm_leq_impl(lsol::LasingSol, mvar_vec::AbsVec{<:LasingModalVar})
    leq = 0.0
    b = lsol.vtemp
    for m = lsol.m_active
        lsd = mvar_vec[m].lsd
        ψ = lsol.ψ[m]
        linapply!(b, lsd, ψ)  # b = A * ψ
        leq = max(leq, norm(b))  # 2-norm for each mode, ∞-norm between modes
    end

    return leq  # return 0.0 if lsol.m_active is empty
end


function norm_leq(lsol::LasingSol, lvar::LasingVar, gp::GainProfile)
    lvar.inited || throw(ArgumentError("lvar is uninitialized: call init_lvar!(...) first."))

    return norm_leq_impl(lsol, lvar.mvar_vec)
end


# Create the mth constraint equation for ∆ω and ∆a.
function set_constraint!(cst::LasingConstraint,
                         lsol::LasingSol,
                         indₘ::Integer,  # index of m in m_active
                         mvar::LasingModalVar,  # modal variables for mth lasing mode
                         rvar::LasingReducedVar,
                         gp::GainProfile)
    # Retrieve necessary variables for constructing the constraint.
    m = lsol.m_active[indₘ]  # modal index of equation of interest
    L = num_active(lsol)  # number of currently lasing modes
    K = length(gp)  # number of atomic classes

    A = cst.A  # left-hand-side matrix of constraint equation
    b = cst.b  # right-hand-side vector of constraint equation

    ω = lsol.ω  # M-vector whose mth entry is current approximate mth eigenfrequency
    a² = lsol.a²  # M-vector whose mth entry is squared amplitude of current approximate mth eigenmode
    ψ = lsol.ψ  # M-vector whose mth entry is current approximate mth eigenmode
    abs2ψ = lsol.abs2ψ  # M-vector whose mth entry is absolute square of current approximate mth eigenmode

    ωₘ = ω[m]
    ωₘ² = ωₘ^2
    ψₘ = ψ[m]
    iₐₘ = lsol.iₐ[m]  # where mth eigenmode is normalized

    vtemp = lsol.vtemp  # scratch space 1
    wtemp = lsol.wtemp  # scratch space 2

    γ = gp.gain  # K-vector whose kth entry is γ function of atomic class k
    abs2γ = gp.abs2gain  # K-vector whose kth entry is |γ|² function of atomic class k
    abs2γ′ = gp.abs2gain′  # K-vector whose kth entry is ∂|γ|²/∂ω function of atomic class k

    D′ = rvar.D′  # K-vector whose kth entry is derivative of population inversion of class-k atoms w.r.t. hole-burning term

    # Calculate the iₐth row of A⁻¹ and keep it as a column vector
    eᵢₐ = vtemp
    eᵢₐ .= 0
    eᵢₐ[iₐₘ] = 1
    r = wtemp  # storage for row vector
    linsolve_transpose!(r, mvar.lsd, eᵢₐ)

    # Construct A and b.  Note that they are initialized to zero outside the present
    # function, by update_lsol_impl!.

    # Set the mth complex row of right-hand-side vector b of the constraint.
    @assert ψₘ[iₐₘ] ≈ 1
    b[2indₘ-1] = 1.0  # real(ψₘ[iₐₘ])
    b[2indₘ] = 0.0  # imag(ψₘ[iₐₘ])

    # Set the mth complex row of the left-hand-side matrix A of the constraint.
    for indⱼ = 1:L
        j = lsol.m_active[indⱼ]  # jth complex column

        ωⱼ = ω[j]
        a²ⱼ = a²[j]
        abs2ψⱼ = abs2ψ[j]

        # Set the coefficient of ∆ωⱼ.
        vtemp .= 0
        ωₘ²a²ⱼ = ωₘ² * a²ⱼ  # scalar
        for k = 1:K
            σ = (ωₘ²a²ⱼ * abs2γ′[k](ωⱼ)) * γ[k](ωₘ)  # scalar: (real * real) * complex
            vtemp .+= σ .* D′[k]
        end
        vtemp .*= ψₘ .* abs2ψⱼ

        # The mth complex row has an additional contribution in the mth complex column from
        # the ωₘ-dependence outside the ω-dependent D.
        j==m && (vtemp .+= mvar.∂f∂ω)

        mω = BLAS.dotu(r, vtemp)
        A[2indₘ-1, 2indⱼ-1] = real(mω)
        A[2indₘ, 2indⱼ-1] = imag(mω)

        # Set the coefficient of ∆a²ⱼ.
        vtemp .= 0
        for k = 1:K
            σ = (ωₘ² * abs2γ[k](ωⱼ)) * γ[k](ωₘ)  # scalar: (real * real) * complex
            vtemp .+= σ .* D′[k]
        end
        vtemp .*= ψₘ .* abs2ψⱼ

        ma² = BLAS.dotu(r, vtemp)
        A[2indₘ-1, 2indⱼ] = real(ma²)
        A[2indₘ, 2indⱼ] = imag(ma²)
    end

    return nothing
end


# Update ψ for the mth mode.  This function does not update ω and a².
function update_ψₘ!(lsol::LasingSol,
                    ∆ωa²::VecFloat,
                    indₘ::Integer,  # index of m in m_active
                    mvar::LasingModalVar,
                    rvar::LasingReducedVar,
                    gp::GainProfile)
    # Retrieve necessary variables for constructing the constraint.
    m = lsol.m_active[indₘ]  # modal index of equation of interest
    L = num_active(lsol)  # number of currently lasing modes
    K = length(gp)  # number of atomic classes

    ω = lsol.ω  # M-vector whose mth entry is current approximate mth eigenfrequency
    a² = lsol.a²  # M-vector whose mth entry is squared amplitude of current approximate mth eigenmode
    ψ = lsol.ψ  # M-vector whose mth entry is current approximate mth eigenmode
    abs2ψ = lsol.abs2ψ  # M-vector whose mth entry is absolute square of current approximate mth eigenmode

    ωₘ = ω[m]
    ωₘ² = ωₘ^2
    ψₘ = ψ[m]
    iₐₘ = lsol.iₐ[m]  # where mth eigenmode is normalized

    vtemp = lsol.vtemp  # scratch space 1
    wtemp = lsol.wtemp  # scratch space 1

    γ = gp.gain  # K-vector whose kth entry is γ function of atomic class k
    abs2γ = gp.abs2gain  # K-vector whose kth entry is |γ|² function of atomic class k
    abs2γ′ = gp.abs2gain′  # K-vector whose kth entry is ∂|γ|²/∂ω function of atomic class k

    D′ = rvar.D′  # K-vector whose kth entry is derivative of population inversion of class-k atoms w.r.t. hole-burning term

    wtemp .= 0  # initialize for mth row
    for indⱼ = 1:L
        j = lsol.m_active[indⱼ]  # jth complex column

        ωⱼ = ω[j]
        a²ⱼ = a²[j]
        abs2ψⱼ = abs2ψ[j]

        ∆ωⱼ = ∆ωa²[2indⱼ-1]
        ∆a²ⱼ = ∆ωa²[2indⱼ]

        # Add the contribution of ∆ωⱼ.
        vtemp .= 0  # initialize for ∆ωⱼ
        ωₘ²a²ⱼ = ωₘ² * a²ⱼ  # scalar
        for k = 1:K
            σ = (∆ωⱼ * ωₘ²a²ⱼ * abs2γ′[k](ωⱼ)) * γ[k](ωₘ)  # scalar: (real * real * real) * complex
            vtemp .+= σ .* D′[k]
        end
        vtemp .*= abs2ψⱼ  # ψₘ will be multiplied outside for loop, because it is common for both ∆ω and ∆a²
        wtemp .+= vtemp

        # Add the contribution of ∆a²ⱼ.
        vtemp .= 0  # initialize for ∆a²ⱼ
        for k = 1:K
            σ = (∆a²ⱼ * ωₘ² * abs2γ[k](ωⱼ)) * γ[k](ωₘ)  # scalar: (real * real * real) * complex
            vtemp .+= σ .* D′[k]
        end
        vtemp .*= abs2ψⱼ  # ψₘ will be multiplied outside for loop, because it is common for both ∆ω and ∆a²
        wtemp .+= vtemp
    end
    wtemp .*= ψₘ

    # The mth complex row has an additional contribution from the ωₘ-dependence outside the
    # ω-dependent D.
    ∆ωₘ = ∆ωa²[2indₘ-1]
    wtemp .+= ∆ωₘ .* mvar.∂f∂ω

    # @info "‖A‖₁ = $(opnorm(mvar.A,1)), ‖vtemp‖ = $(norm(vtemp))"
    linsolve!(ψₘ, mvar.lsd, wtemp)  # ψₘ .= mvar.lsd \ wtemp

    # Normalize ψₘ just in case ψₘ[iₐₘ] ≠ 1.
    # ψₘ[iₐₘ]≈1 || @warn "lasing mode m = $m is slightly nonnormal: |ψₘ[iₐₘ]-1| = $(abs(ψₘ[iₐₘ]-1)).  The mode will be renormalized."
    # @info "ψₘ[iₐₘ] = $(ψₘ[iₐₘ])"
    ψₘ ./= ψₘ[iₐₘ]

    return nothing
end


# lvar must be already initialized by init_lvar! before starting the fixed-point iteration.
function update_lsol!(lsol::LasingSol, lvar::LasingVar, gp::GainProfile)
    lvar.inited || throw(ArgumentError("lvar is uninitialized: call init_lvar!(...) first."))
    update_lsol_impl!(lsol, lvar.mvar_vec, lvar.rvar, lvar.cst, gp)
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
                           mvar_vec::AbsVec{<:LasingModalVar},  # must be already initialized
                           rvar::LasingReducedVar,  # must be already initialized
                           cst::LasingConstraint,
                           gp::GainProfile)
    # Construct the constraint equation on ∆ω and ∆a.
    L = num_active(lsol)

    A = cst.A
    b = cst.b
    ∆ωa² = cst.∆ωa²

    A .= 0
    b .= 0
    ∆ωa² .= 0

    # Below, we iterate over the contiguous indices indₘ to retrieve the currently lasing
    # mode index m from m_active, instead of directly iterating over m_active.  This is
    # because we cannot use a submatrix @view(A[ind,ind]) in ldiv! when ind is not a
    # contiguous range of indices.  See https://github.com/JuliaLang/julia/issues/30097.
    for indₘ = 1:L
        m = lsol.m_active[indₘ]
        set_constraint!(cst, lsol, indₘ, mvar_vec[m], rvar, gp)
    end

    # Calculate ∆ω and ∆a.
    LU = lu!(@view(A[1:2L,1:2L]))
    ldiv!(@view(∆ωa²[1:2L]), LU, @view(b[1:2L]))
    # @info "A = $(A[1:2L,1:2L]), b = $(b[1:2L]), ∆ωa² = $(∆ωa²[1:2L])"

    # Update ψ.
    # @info "lsol.ω = $(lsol.ω), lsol.a² = $(lsol.a²), lsol.m_active = $(lsol.m_active)"
    for indₘ = 1:L
        m = lsol.m_active[indₘ]
        # @info "before apply: ‖lsol.ψ[$m]‖ = $(norm(lsol.ψ[m]))"
        update_ψₘ!(lsol, ∆ωa², indₘ, mvar_vec[m], rvar, gp)
        # @info "after apply: ‖lsol.ψ[$m]‖ = $(norm(lsol.ψ[m]))"
    end

    # Update ω and a².
    #
    # abs2ψ is not updated here, because the Anderson acceleration affinely combines the
    # previous solutions and therefore makes abs2ψ for ψ generated by fixed-point function
    # irrelevant.  Therefore, abs2ψ must be updated at the beginning of the fixed-point
    # function (where abs2ψ is used for the first time in that iteration step) rather than
    # at the end.  This is inside init_lvar!, whose evaluation in checked at the beginning
    # of update_lsol!.
    #
    # The following loop must not be merged with the loop above, because update_ψₘ! uses
    # un-updated ω and a².
    for indₘ = 1:L
        m = lsol.m_active[indₘ]

        ∆ωₘ = ∆ωa²[2indₘ-1]
        ∆a²ₘ = ∆ωa²[2indₘ]

        lsol.ω[m] += ∆ωₘ
        lsol.a²[m] += ∆a²ₘ
    end

    return nothing
end
