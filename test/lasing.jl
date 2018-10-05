@testset "lasing" begin

# Create a system.
Ngrid = [3,3,3]
N = 3prod(Ngrid)
ind_in = 5:7
εc = ones(ComplexF64, N)
εc[ind_in] .= 12+0.1im
m = 1
ωₘ = 100.0
ω₀ = 1.0
γperp = 1.0
D₀ = fill(0.01, N)
D₀[ind_in] .= 1.0
gp = GainProfile(ω₀, γperp, D₀)

# Create a solution.
M = 3
ω = rand(M)
ω[m] = ωₘ
a² = rand(M)
abs2gain = gen_abs2γ(ω₀,γperp)
abs2gain′ = gen_abs2γ′(ω₀,γperp)
Ψ = randn(ComplexF64,N,M)
ψ = [Ψ[:,j] for j = 1:M]
iarray = rand(1:N, M)
vtemp = similar(ψ[1])
sol = LasingSol(ω, a², ψ, iarray, vtemp)  # ω = rand(M); this is fine because init_reduced_var does not use ω
sol.active = fill(true, M)
sol.m_active = collect(1:M)

# Create ∆solution.
∆a² = rand(M)
∆ω = rand(M)
∆Ψ = randn(ComplexF64,N,M)
∆ψ = [∆Ψ[:,j] for j = 1:M]
∆sol = SALTBase.∆LasingSol(∆ω, ∆a², ∆ψ, vtemp)

# Check reduced variables.
hb = Vector{Float64}(undef, N)
hole_burning!(hb, ω, a², ψ, abs2gain)
D = D₀ ./ hb
D′ = -D₀ ./ abs2.(hb)
∆D = similar(D)
SALTBase.∆popinv!(∆D, D′, ∆sol, sol, gp)
∇ₐ₂D = [zeros(N) for m = 1:M]
SALTBase.∇ₐ₂popinv!(∇ₐ₂D, D′, sol, gp)
∇ωD = [zeros(N) for m = 1:M]
SALTBase.∇ωpopinv!(∇ωD, D′, sol, gp)
rvar = SALTBase.LasingReducedVar(N, M)
SALTBase.init_reduced_var!(rvar, ∆sol, sol, gp)
@testset "reduced variables" begin
    @test rvar.D ≈ D
    @test rvar.D′ ≈ D′
    @test ∆D ≈ D′ .* (real.(conj.(Ψ) .* ∆Ψ) * (2 .* a² .* abs2gain.(ω)))
    @test rvar.∆D ≈ ∆D
    ∇ₐ₂D_mat = D′ .* (abs2.(Ψ) .* abs2gain.(ω)')
    @test ∇ₐ₂D ≈ [∇ₐ₂D_mat[:,j] for j = 1:M]
    @test rvar.∇ₐ₂D ≈ ∇ₐ₂D
    ∇ωD_mat = D′ .* (abs2.(Ψ) .* (a² .* abs2gain′.(ω))')
    @test ∇ωD ≈ [∇ωD_mat[:,j] for j = 1:M]
    @test rvar.∇ωD ≈ ∇ωD
end


mvar = SALTBase.LasingModalVar(DefaultLSD(), N)
SALTBase.init_modal_var!(mvar, m, sol, rvar, gp, εc)

γ = gen_γ(ω₀, γperp)(ωₘ)
γ′ = gen_γ′(ω₀, γperp)(ωₘ)
ε = εc + γ*D
@testset "modal variables" begin
    @test mvar.ω²γψ ≈ (ωₘ^2 * γ) .* ψ[m]
    @test mvar.∂f∂ω ≈ (2ωₘ*ε + ωₘ^2*γ′*D) .* ψ[m]
end

end  # @testset "lasing"





# @test mvar.rowA⁻¹ᵢₐ ≈ transpose(transpose(eᵢₐ)/A)
