@testset "salteq" begin

# Create a system.
Ngrid = [3,3,3]
N = 3prod(Ngrid)
ind_in = 5:7
εc = ones(ComplexF64, N)
εc[ind_in] .= 12+0.1im
m = 1
ω₀ = 1.0
γperp = 1.0
D₀ = fill(0.01, N)
D₀[ind_in] .= 1.0

M = 4
ω = rand(M)
a² = rand(M)
Ψ = randn(ComplexF64,N,M)
ψ = [Ψ[:,j] for j = 1:M]

gain = gen_γ(ω₀, γperp)
gain′ = gen_γ′(ω₀, γperp)
abs2gain = gen_abs2γ(ω₀, γperp)
abs2gain′ = gen_abs2γ′(ω₀, γperp)
@test gain′.(ω) ≈ -gain.(ω).^2 ./ γperp
@test abs2gain.(ω) ≈ abs2.(gain.(ω))
@test abs2gain′.(ω) ≈ -2 .* abs2gain.(ω).^2 .* (ω.-ω₀) ./ γperp^2

hb = Vector{Float64}(undef, N)
hole_burning!(hb, ω, a², ψ, abs2gain)
@test all(hb .> 1)
@test hb ≈ 1 .+ abs2.(Ψ) * (abs2gain.(ω) .* a²)

end  # @testset "salteq"
