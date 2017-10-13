using SALT3D
using PyPlot

# Set up the problem
N = 100
Mmax = M = m = 1
Lcavity = 1.
hx = Lcavity / (N + 1)
x = linspace(hx, Lcavity-hx, N)

ωn(n) = (2/hx) * sin(n*π*hx / (2*Lcavity))
indmode = 2
assert(0 < indmode <= N)  # The discretized Laplacian is N × N, so the system has N eigenvalues
ωₐ = ωn(indmode)  # (arbitrary choice of) transition frequency
γ⟂ = 1  # relaxation rate
σ = 0.03  # imaginary part of εc
εc = fill(1 + im * σ, N)  # loss is introduced by material, not by radiation
d = 1.02
D₀ = σ .* fill(d, N)
param = SALTParam(ωₐ, γ⟂, εc, D₀)

# Choose a guess solution.
diags = (1/hx^2) * -2 * ones(N)
off_diags = (1/hx^2) * 1 * ones(N-1)
CC = -sparse(SymTridiagonal(diags, off_diags))  # negative Laplacian = positive curl of curl

Aₙₗ = spdiagm(εc) \ copy(CC)
Ω², Ψ = eigs(Aₙₗ, nev=1, sigma=ωₐ^2)
ωguess = real(√Ω²[1])
ψguess = Ψ[:,1]
imax = indmax(abs.(ψguess))
aguess = abs(ψguess[imax])
ψguess = ψguess ./ aguess
lsol = LasingSol([ωguess], [aguess], [ψguess], [imax])

# Create variables.
lvar = LasingVar(CC, M)

# Procedure
# - Initialize ∆lsol.  (This is a guess ∆lsol to feed to the fixed-point equation solver.)
# - Initialize rvar and mvar for a given lsol and ∆lsol.
# - Feed the initialized ∆lsol to update_∆lsol.
# - Move lsol by ∆lsol.

for k = 1:10
    init_lvar!(lvar, lsol, CC, param)
    normleq = norm_leq(lsol, lvar)
    info("‖leq‖ = $normleq")
    if normleq ≤ Base.rtoldefault(Float64)
        break
    end
    update_lsol!(lsol, lvar, CC, param)
end

using PyPlot
plot(1:N, abs.(ψguess), "ro", 1:N, abs.(lsol.ψ[m]), "b-")

println("a = $(lsol.a[m])")  # must be around 0.163459
