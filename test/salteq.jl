using MaxwellFDM

@testset "salteq" begin

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

M = 4
a² = rand(M)
Ψ = randn(N,M) + im .* randn(N,M)
ψ = [Ψ[:,j] for j = 1:M]

γ = gain(ωₘ, ωₐ, γ⟂)
γ′ = SALTBase.gain′(ωₘ, ωₐ, γ⟂)
@test γ′ ≈ -γ^2 / γ⟂

hb = Vector{Float64}(N)
hole_burning!(hb, a², ψ)
@test all(hb .> 1)
@test hb ≈ 1 + abs2.(Ψ) * a²

end  # @testset "salteq"
