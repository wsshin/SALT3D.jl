# Define functions to find ∆ω, ∆a, ∆ψ by the implicit Newton step calculation algorithm.
# Note that ω, a, ψ are fixed in this calculation.

# We can save computations by implementing the assumption ∆D = 0, and also by updating ψ
# directly without calculating ∆ψ.

export LasingSol, LasingVar
export norm_leq, update_lsol!, fixedpt!

# Solutions to the lasing equation.
mutable struct LasingSol{VC<:AbsVecComplex}  # VC can be PETSc vector
    ω::VecFloat  # M real numbers: frequencies of modes (M = # of lasing modes)
    a²::VecFloat  # M real numbers: squared "amplitudes" of modes
    ψ::Vector{VC}  # M complex vectors: normalized modes
    iₐ::VecInt  # M integers: row indices where amplitudes are measured
    act::VecBool  # act[m] is true if mode m is active (i.e., lasing)
    m_act::VecInt  # vector of active (i.e., lasing) mode indices; collection of m such that act[m] == true
    function LasingSol{VC}(ω::AbsVecReal,
                           a²::AbsVecReal,
                           ψ::AbsVec{<:AbsVecNumber},
                           iₐ::AbsVecInteger) where {VC<:AbsVecComplex}
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

        return new(ω, a², ψ, iₐ, fill(false,M), VecInt(0))
    end
end
LasingSol(ω::AbsVecReal, a²::AbsVecReal, ψ::AbsVec{VC}, iₐ::AbsVecInteger) where {VC<:AbsVecComplex} =
    LasingSol{VC}(ω, a², ψ, iₐ)

# To do: check if the following works for vtemp of PETSc vector type.
LasingSol(vtemp::AbsVec,  # template vector with N entries
          M::Integer) =
    LasingSol(zeros(M), zeros(M), [similar(vtemp,CFloat).=0 for m = 1:M], zeros(Int,M))
LasingSol(N::Integer, M::Integer) = LasingSol(VecFloat(N), M)

Base.length(lsol::LasingSol) = length(lsol.ψ)

function Base.normalize!(lsol::LasingSol)
    for m = lsol.m_act
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
∆LasingSol(vtemp::AbsVec,  # template vector with N entries
           M::Integer) =
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
LasingReducedVar(vtemp::AbsVec,  # template vector with N entries
                 M::Integer) =
    LasingReducedVar(similar(vtemp,Float), similar(vtemp,Float), similar(vtemp,Float), [similar(vtemp,Float) for m = 1:M])
LasingReducedVar(N::Integer, M::Integer) =  LasingReducedVar(VecFloat(N), M)


# Lasing equation variables that are unique to each lasing mode
mutable struct LasingModalVar{MC<:AbsMatComplex,VC<:AbsVecComplex}  # MC and VC can be PETSc matrix and vector
    # Consider adding ε as a field, in case the user wants to solve the "linear" eigenvalue
    # equation for some reason.
    A::MC
    ω²γψ::VC
    ∂f∂ω::VC
    function LasingModalVar{MC,VC}(A::AbsMatNumber,  # dense A is automatically converted to sparse matrix if MC is sparse type
                                   ω²γψ::AbsVecNumber,
                                   ∂f∂ω::AbsVecNumber) where {MC<:AbsMatComplex,VC<:AbsVecComplex}
        length(ω²γψ)==length(∂f∂ω) ||
            throw(ArgumentError("length(ω²γψ) = $(length(ω²γψ)) and length(∂f∂ω) = $(length(∂f∂ω)) must be the same."))
        N = length(ω²γψ)
        size(A) == (N,N) || throw(ArgumentError("Each entry of size(A) = $(size(A)) and length(ω²γψ) = $N must be the same."))

        return new(A, ω²γψ, ∂f∂ω)
    end
end
LasingModalVar(A::MC, ω²γψ::VC, ∂f∂ω::VC) where {MC<:AbsMatComplex,VC<:AbsVecComplex} =
    LasingModalVar{MC,VC}(A, ω²γψ, ∂f∂ω)

# To do: check if the following works for vtemp of PETSc vector type.
LasingModalVar(mtemp::AbsMat,  # template N×N matrix (e.g., sparse matrix with all nonzero locations already specified)
               vtemp::AbsVec) =  # template vector with N entries
    LasingModalVar(similar(mtemp,CFloat), similar(vtemp,CFloat), similar(vtemp,CFloat))
LasingModalVar(mtemp::AbsMat) =
    (N = size(mtemp)[1]; LasingModalVar(similar(mtemp,CFloat), VecComplex(N), VecComplex(N)))


mutable struct LasingConstraint
    A::MatFloat  # constraint matrix
    b::VecFloat  # constraint vector
    m2_act::VecBool  # vector of 2M booleans; true if corresponding row and column are active
    LasingConstraint(M::Integer) = new(MatFloat(2M,2M), VecFloat(2M), VecBool(2M))
end

function activate!(cst::LasingConstraint, lsol::LasingSol)
    M = length(lsol)
    for m = 1:M
        cst.m2_act[2m-1] = cst.m2_act[2m] = lsol.act[m]
    end
end


# Calculate the change induced in population inversion by the change in ψ's.
# Note that this is the only function in this file whose output depends on ∆ψ's.
function ∆popinv!(∆D::AbsVecFloat,  # output
                  D′::AbsVecReal,  # derivative of population inversion; output of popinv′
                  ∆lsol::∆LasingSol,
                  lsol::LasingSol)
    ∆D .= 0  # initialize
    for m = lsol.m_act
        ∆D .+= 2lsol.a²[m] .* real.(conj.(lsol.ψ[m]) .* ∆lsol.∆ψ[m])
    end
    ∆D .*= D′

    return nothing
end


# Calculate the gradient of population inversion with respect to amplitudes a.
function ∇ₐ₂popinv!(∇ₐ₂D::AbsVec{<:AbsVecFloat},  # output
                    D′::AbsVecReal,  # derivative of population inversion; output of popinv′
                    lsol::LasingSol)
    for m = lsol.m_act
        ∇ₐ₂D[m] .= abs2.(lsol.ψ[m]) .* D′
    end

    return nothing
end


# Components necessary for fixed-point iteration
mutable struct LasingVar{MC<:AbsMatComplex,VC<:AbsVecComplex,VF<:AbsVecFloat}
    ∆lsol::∆LasingSol{VC}
    mvar_vec::Vector{LasingModalVar{MC,VC}}
    rvar::LasingReducedVar{VF}
    cst::LasingConstraint
end
LasingVar(mtemp::AbsMat, vtemp::AbsVec, M::Integer) =
    LasingVar(∆LasingSol(vtemp, M), [LasingModalVar(mtemp, vtemp) for m = 1:M], LasingReducedVar(vtemp, M), LasingConstraint(M))
LasingVar(mtemp::AbsMat, M::Integer) = (N = size(mtemp)[1]; LasingVar(mtemp, VecFloat(N), M))


function init_reduced_var!(rvar::LasingReducedVar, ∆lsol::∆LasingSol, lsol::LasingSol, param::SALTParam)
    hole_burning!(rvar.D′, lsol.a², lsol.ψ)  # temporarily store hole-burning term in D′

    rvar.D .= param.D₀ ./ rvar.D′  # D = D₀ / (1 + ∑a²|ψ|²)
    rvar.D′ .= -param.D₀ ./ abs2.(rvar.D′)  # D′(∑a²|ψ|²) = -D₀ / (1+∑a²|ψ|²)²

    ∆popinv!(rvar.∆D, rvar.D′, ∆lsol, lsol)  # comment this out when ∆ψ_old = 0 (so that ∆D = 0)
    ∇ₐ₂popinv!(rvar.∇ₐ₂D, rvar.D′, lsol)

    # info("‖D‖ = $(norm(rvar.D)), ‖D′‖ = $(norm(rvar.D′)), ‖∆D‖ = $(norm(rvar.∆D))")
    # for m = lsol.m_act
    #     info("‖∇ₐ₂D[$m]‖ = $(norm(rvar.∇ₐ₂D[m]))")
    # end

    return nothing
end


function init_modal_var!(mvar::LasingModalVar,
                         m::Integer,  # index of lasing mode
                         lsol::LasingSol,
                         rvar::LasingReducedVar,
                         CC::AbsMatNumber,
                         param::SALTParam)
    isreal(lsol.ω[m]) || throw(ArgumentError("lsol.ω[$m] = $(lsol.ω[m]) must be real."))
    ω = real(lsol.ω[m])

    γ = gain(ω, param.ωₐ, param.γ⟂)
    γ′ = gain′(ω, param.ωₐ, param.γ⟂)

    # Below, avoid allocations and use preallocated arrays in mvar.
    ε = mvar.∂f∂ω  # temporary storage for effective permitivity: εc + γ D
    ε .= param.εc .+ γ .* rvar.D

    # Move A, rowA⁻¹ᵢₐ away.  These need to be used only
    create_A!(mvar.A, CC, ω, ε)
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


function init_lvar!(lvar::LasingVar, lsol::LasingSol, CC::AbsMatNumber, param::SALTParam)
    ∆lsol = lvar.∆lsol
    mvar_vec = lvar.mvar_vec
    rvar = lvar.rvar

    init_∆lsol!(∆lsol)  # make ∆lsol all zero
    init_reduced_var!(rvar, ∆lsol, lsol, param)
    for m = lsol.m_act
        init_modal_var!(mvar_vec[m], m, lsol, rvar, CC, param)
    end

    return nothing
end


function norm_leq(lsol::LasingSol, mvar_vec::AbsVec{<:LasingModalVar})
    leq² = 0.0
    for m = lsol.m_act
        A = mvar_vec[m].A
        ψ = lsol.ψ[m]
        leq² = max(leq², sum(abs2,A*ψ))  # 2-norm for each mode, 1-norm between modes
    end

    return √leq²  # return 0.0 if lsol.m_act is empty
end

function norm_leq(lsol::LasingSol, lvar::LasingVar, CC::AbsMatNumber, param::SALTParam)
    # Call init_lvar!, which is necessary for using update_lsol!, here in order to force
    # checking the norm before using update_lsol!.
    init_lvar!(lvar, lsol, CC, param)

    return norm_leq(lsol, lvar.mvar_vec)
end

# To do: use an iterative solver inside this, and pass a storage for the output.  (We need
# to change init_modal_var! accordingly.)
function row_A⁻¹(iₐ::Integer,  # row index
                 A::AbsMatNumber)  # matrix A
    N = size(A)[1]
    eᵢₐ = zeros(N)
    eᵢₐ[iₐ] = 1
    # info("‖A‖₁ = $(norm(A,1))")
    rowA⁻¹ᵢₐ = A.' \ eᵢₐ  # R = A⁻ᵀ; rowA⁻¹ᵢₐ = (column form of iₐth row of A⁻¹) = (eᵢₐᵀ A⁻¹)ᵀ = A⁻ᵀ eᵢₐ

    return rowA⁻¹ᵢₐ
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

    # Retrieve necessary variables for constructing the constraint.
    ∆D = rvar.∆D
    ∇ₐ₂D = rvar.∇ₐ₂D

    r = row_A⁻¹(iₐ, mvar.A)

    ω²γψ = mvar.ω²γψ
    ∂f∂ω = mvar.∂f∂ω

    A = cst.A  # constraint right-hand-side matrix
    b = cst.b  # constraint right-hand-side vector
    vtemp = ∆lsol.vtemp

    # Set the right-hand-side vector of the constraint.
    vtemp .= ∆D.*ω²γψ
    ζv = ψ[iₐ] - BLAS.dotu(r, vtemp)  # scalar; note negation because ζv is quantity on RHS
    # info("ψ[iₐ] = $(ψ[iₐ]), ‖ψ‖ = $(norm(ψ))")
    # ζv = ψ[iₐ]  # ∆D = 0
    b[2m-1] = real(ζv)
    b[2m] = imag(ζv)

    # Set the left-hand-side matrix of the constraint.
    ζω = BLAS.dotu(r, ∂f∂ω)
    A[2m-1,2m-1] = real(ζω)
    A[2m,2m-1] = imag(ζω)

    for j = lsol.m_act
        vtemp .= ∇ₐ₂D[j] .* ω²γψ  # this uses no allocations
        ζa² = BLAS.dotu(r, vtemp)
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
    vtemp = ∆lsol.vtemp
    vtemp .= (∆D .* ω²γψ) .+ (∆ω .* ∂f∂ω)
    # vtemp .= (∆ω .* ∂f∂ω)  # ∆D = 0
    for j = lsol.m_act
        # info("‖∇ₐ₂D[$j]‖ = $(norm(∇ₐ₂D[j]))")
        vtemp .+= ∆a²[j] .* (∇ₐ₂D[j] .* ω²γψ)
    end

    # # Calculate ∆ψ.
    # ∆ψ .= mvar.A \ vtemp
    # ∆ψ .-= ψ
    # # ∆ψ[lsol.iₐ[m]] = 0

    # info("‖A‖₁ = $(norm(mvar.A,1)), ‖vtemp‖ = $(norm(vtemp))")

    # ψ .= mvar.A \ vtemp
    ### Pardiso begins.
    ps = PardisoSolver()
    set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)

    # First set the matrix type to handle general real symmetric matrices
    T = eltype(mvar.A)
    if T<:Real
        set_matrixtype!(ps, Pardiso.REAL_NONSYM)
    else  # T<:Complex
        set_matrixtype!(ps, Pardiso.COMPLEX_NONSYM)
    end

    # Initialize the default settings with the current matrix type
    pardisoinit(ps)

    # Get the correct matrix to be sent into the pardiso function.
    # :N for normal matrix, :T for transpose, :C for conjugate
    A_pardiso = get_matrix(ps, mvar.A, :N)

    # Analyze the matrix and compute a symbolic factorization.
    set_phase!(ps, Pardiso.ANALYSIS)
    set_perm!(ps, randperm(size(mvar.A, 1)))
    pardiso(ps, A_pardiso, vtemp)

    # Compute the numeric factorization.
    set_phase!(ps, Pardiso.NUM_FACT)
    pardiso(ps, A_pardiso, vtemp)

    # Compute the solutions X using the symbolic factorization.
    set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    # set_solver!(ps, Pardiso.ITERATIVE_SOLVER)
    pardiso(ps, ψ, A_pardiso, vtemp)
    ### Pardiso ends.

    iₐ = lsol.iₐ[m]
    ψ ./= ψ[iₐ]

    # The following could have been updated before this function, because they were already
    # prepared.
    lsol.ω[m] += ∆ω
    lsol.a²[m] += ∆a²[m]

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
function update_lsol!(lsol::LasingSol,
                      ∆lsol::∆LasingSol,
                      mvar_vec::AbsVec{<:LasingModalVar},  # must be already initialized
                      rvar::LasingReducedVar,  # must be already initialized
                      cst::LasingConstraint,
                      param::SALTParam)
    # Construct the constraint equation on ∆ω and ∆a.
    cst.A .= 0
    cst.b .= 0
    for m = lsol.m_act
        set_constraint!(cst, ∆lsol, lsol, m, mvar_vec[m], rvar)
    end

    # Calculate ∆ω and ∆a.
    activate!(cst, lsol)
    ind = cst.m2_act
    ∆ωa² = cst.A[ind,ind] \ cst.b[ind]
    # info("cst.A = $(cst.A[ind,ind]), cst.b = $(cst.b[ind]), ∆ωa² = $∆ωa²")
    c = 0  # count
    for m = lsol.m_act
        c += 1
        ∆lsol.∆ω[m] = ∆ωa²[2c-1]
        ∆lsol.∆a²[m] = ∆ωa²[2c]
    end

    # Update ∆ψ.
    # info("lsol.ω = $(lsol.ω), lsol.a² = $(lsol.a²), lsol.m_act = $(lsol.m_act)")
    for m = lsol.m_act
        # info("before apply: ‖lsol.ψ[$m]‖ = $(norm(lsol.ψ[m]))")
        apply_∆solₘ!(lsol, ∆lsol, m, mvar_vec[m], rvar)
        # info("after apply: ‖lsol.ψ[$m]‖ = $(norm(lsol.ψ[m]))")
    end

    return nothing
end


# lvar must be already initialized by init_lvar! before starting the fixed-point iteration.
update_lsol!(lsol::LasingSol, lvar::LasingVar, CC::AbsMatNumber, param::SALTParam) =
    update_lsol!(lsol, lvar.∆lsol, lvar.mvar_vec, lvar.rvar, lvar.cst, param)


# To use andersonaccel!, implement anderson_SALT! that accepts SALTSol as an initial guess
# and g! that takes SALTSol and returns SALTSol.  anderson_SALT! must create a version of g!
# that takes and returns vectors using CatViews.

# I will need to implement `reinterpret` for PETSc vectors to view complex PETSc vectors as
# a real PETSc vector.

function lsol2rvec(lsol::LasingSol)
    m_act = lsol.m_act
    ψr = reinterpret.(Float, lsol.ψ[lsol.m_act])
    # ψr = lsol.ψ[lsol.m_act]  # complex version

    return CatView(lsol.ω[lsol.m_act], lsol.a²[lsol.m_act], ψr...)
    # return CatView(ψr...)
    # return CatView(lsol.ω[lsol.m_act], lsol.a²[lsol.m_act])
end


# Notes on applying andersonaccel! to SALT:
# - First, note that andersonaccel! takes g!(y,x) that updates y without changing x.  On the
# other hand, our update_lsol! updates the solution in-place.  Therefore, we need to think
# about how to write the fixed-point equation properly.
# - Or, we could change andersonaccel! such that it takes g!(x) that updates x in-place.


# Used with anderson_steven.jl in usage1d_slab_multimode_anderson_orig.jl.
# function fixedpt!(y, x, lsol::LasingSol, lvar::LasingVar, CC::AbsMatNumber, param::SALTParam)
#     rvec = lsol2rvec(lsol)
#     # info("size(x) = $(size(x)), size(rvec) = $(size(rvec))")
#     rvec .= x
#
#     init_lvar!(lvar, lsol, CC, param)
#     update_lsol!(lsol, lvar, CC, param)
#     y .= rvec
# end
