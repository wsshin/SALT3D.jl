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
a = rand(M)
Ψ = randn(N,M) + im .* randn(N,M)
ψ = [Ψ[:,j] for j = 1:M]

γ = gain(ωₘ, ωₐ, γ⟂)
γ′ = gain′(ωₘ, ωₐ, γ⟂)
@test γ′ ≈ -γ^2 / γ⟂

hb = Vector{Float64}(N)
hole_burning!(hb, a, ψ)
@test all(hb .> 1)
@test hb ≈ 1 + abs2.(Ψ) * a.^2

# Create A.
Ce = create_curl(PRIM, Ngrid)
Ch = create_curl(DUAL, Ngrid)
CC = Ch*Ce

D = D₀ ./ hb
ε = εc + γ*D
A = similar(CC, Complex128)
create_A!(A, CC, ωₘ, ε)
@test A ≈ CC - ωₘ^2 * spdiagm(ε)

end  # @testset "salteq"
