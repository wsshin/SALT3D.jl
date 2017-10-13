using SALT3D
using MaxwellFD3D
using PyPlot

Mmax = M = m = 1

# Create a grid.
L₀ = 1e-9
unit = PhysUnit(L₀)

Npml = ([0,0,0], [0,0,0])

Nx = 100
Lcavity = 1.
hx = Lcavity / Nx
xprim = collect(linspace(0, Lcavity, Nx+1))
N = 3Nx*1*1  # 3 means three Cartesian components


g3 = Grid(unit, (xprim,[0,1],[0,1]), Npml, [BLOCH,BLOCH,BLOCH])


# Create the equation and solve it.
∆lprim = g3.∆l[nPR]
∆ldual = g3.∆l[nDL]
ebc =  g3.ebc
e⁻ⁱᵏᴸ = ones(3)

Cu = create_curl(PRIM, g3.N, ∆ldual, ebc, e⁻ⁱᵏᴸ, reorder=true)
Cv = create_curl(DUAL, g3.N, ∆lprim, ebc, e⁻ⁱᵏᴸ, reorder=true)
CC = Cv * Cu


# Set up the problem
ωn(n) = (2/hx) * sin(n*π*hx / (2*Lcavity))
indmode = 2
ωₐ = ωn(indmode)  # (arbitrary choice of) transition frequency
γ⟂ = 1  # relaxation rate
σ = 0.03  # imaginary part of εc
εc = fill(1 + im * σ, N)  # loss is introduced by material, not by radiation
d = 1.02
D₀ = σ .* fill(d, N)
param = SALTParam(ωₐ, γ⟂, εc, D₀)

# Choose a guess solution.
Aₙₗ = spdiagm(εc) \ copy(CC)  # A for nonlasing mode
Ω², Ψ = eigs(Aₙₗ, nev=1, sigma=ωₐ^2)
ωguess = real(√Ω²[1])
ψguess = Ψ[:,1]
imax = indmax(abs.(ψguess))
w = mod1(imax, 3)  # Cartesian component that is strongest
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

for k = 1:20
    init_lvar!(lvar, lsol, CC, param)
    normleq = norm_leq(lsol, lvar)
    info("‖leq‖ = $normleq")
    if normleq ≤ Base.rtoldefault(Float64)
        break
    end
    update_lsol!(lsol, lvar, CC, param)
end

using PyPlot
ψwguess = view(ψguess, w:3:endof(ψguess))
ψ = lsol.ψ[m]
ψw = view(ψ, w:3:endof(ψ))
clf()
plot(1:Nx, abs.(ψwguess), "ro", 1:Nx, abs.(ψw), "b-")

println("a = $(lsol.a[m])")  # must be around 0.163459


# ψx = view(ψ, 1:3:endof(ψ))
# ψy = view(ψ, 2:3:endof(ψ))
# ψz = view(ψ, 3:3:endof(ψ))
# ψabs = .√(abs2.(ψx) .+ abs2.(ψy) .+ abs2.(ψz))
# plot(1:Nx, abs.(ψwguess), "ro", 1:Nx, ψabs, "b-")
