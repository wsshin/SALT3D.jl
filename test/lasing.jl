using MaxwellFDM
@testset "step" begin

# Create a system.
Ngrid = [3,3,3]
N = 3prod(Ngrid)
ind_in = 5:7
εc = ones(Complex128, N)
εc[ind_in] .= 12+0.1im
m = 1
ωₘ = 100.0
ωₐ = 1.0
γ⟂ = 1.0
D₀ = fill(0.01, N)
D₀[ind_in] .= 1.0
param = SALTParam(ωₐ, γ⟂, εc, D₀)

# Create a solution.
M = 3
ω = rand(M)
ω[m] = ωₘ
a² = rand(M)
Ψ = randn(N,M) + im .* randn(N,M)
ψ = [Ψ[:,j] for j = 1:M]
iarray = rand(1:N, M)
sol = LasingSol(ω, a, ψ, iarray)  # ω = rand(M); this is fine because init_reduced_var does not use ω

# Create ∆solution.
∆a = rand(M)
∆ω = rand(M)
∆Ψ = randn(N,M) + im .* randn(N,M)
∆ψ = [∆Ψ[:,j] for j = 1:M]
∆sol = SALT3D.∆LasingSol(∆ω, ∆a, ∆ψ)

# Check reduced variables.
hb = Vector{Float64}(N)
hole_burning!(hb, a, ψ)
D = D₀ ./ hb
D′ = -D₀ ./ abs2.(hb)
∆D = similar(D)
SALT3D.∆popinv!(∆D, D′, a, ψ, ∆ψ)
∇ₐD = [zeros(N) for m = 1:M]
SALT3D.∇ₐ₂popinv!(∇ₐD, D′, a², ψ)
rvar = SALT3D.LasingReducedVar(N, M)
SALT3D.init_reduced_var!(rvar, ∆sol, sol, param)
@testset "reduced variables" begin
    @test rvar.D ≈ D
    @test rvar.D′ ≈ D′
    @test ∆D ≈ D′ .* (real.(conj.(Ψ[:,1:M]) .* ∆Ψ) * 2abs2.(a[1:M]))
    @test rvar.∆D ≈ ∆D
    ∇ₐD_mat = D′ .* (abs2.(Ψ[:,1:M]) .* 2a[1:M].')
    @test ∇ₐD ≈ [∇ₐD_mat[:,j] for j = 1:M]
    @test rvar.∇ₐD ≈ ∇ₐD
end

# Check modal variables.
Ce = create_curl(PRIM, Ngrid)
Ch = create_curl(DUAL, Ngrid)
CC = Ch*Ce

mvar = SALT3D.LasingModalVar(CC)
SALT3D.init_modal_var!(mvar, m, sol, rvar, CC, param)

γ = gain(ωₘ, ωₐ, γ⟂)
γ′ = gain′(ωₘ, ωₐ, γ⟂)
ε = εc + γ*D
A = CC - ωₘ^2 * spdiagm(ε)
@testset "modal variables" begin
    @test mvar.A ≈ A
    @test mvar.ω²γψ ≈ (ωₘ^2 * γ) .* ψ[m]
    @test mvar.∂f∂ω ≈ (2ωₘ*ε + ωₘ^2*γ′*D) .* ψ[m]
end

# eᵢₐ = zeros(N)
# eᵢₐ[iarray[m]] = 1


end  # @testset "step"





# @test mvar.rowA⁻¹ᵢₐ ≈ (eᵢₐ.'/A).'
