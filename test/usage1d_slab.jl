using SALT3D
using MaxwellFD3D
using PyPlot

Mmax = 1  # number of modes of interest

# Create a grid.
L₀ = 1e-3  # 1 mm
unit = PhysUnit(L₀)

Npml = ([0,0,0], [10,0,0])

Nx = 120
L = 0.12  # 0.12 mm.  Lcavity = 0.1mm
hx = L / Nx
xprim = collect(linspace(0, L, Nx+1))
N = 3Nx*1*1  # 3 means three Cartesian components

g3 = Grid(unit, (xprim,[0,1],[0,1]), Npml, [PPC,BLOCH,BLOCH])

# Set up the problem
ωₐ = 100
γ⟂ = 40  # relaxation rate

param = SALTParam(3Nx)
param.ωₐ = ωₐ
param.γ⟂ = γ⟂

εslab = 1.2
εc_array = ones(3, Nx)  # loss is introduced by material, not by radiation
εc_array[:,1:100] .= εslab
param.εc .= εc_array[:]

# Create the equation and solve it.
∆lprim = g3.∆l[nPR]
∆ldual = g3.∆l[nDL]
ebc =  g3.ebc
e⁻ⁱᵏᴸ = ones(3)

s = gen_stretch_factor(-ωₐ, g3.l, g3.lpml, g3.Lpml)  # note the minus sign in frequency
sprim = s[nPR]
sdual = s[nDL]

s∆lprim = map((x,y)->x.*y, sprim, ∆lprim)
s∆ldual = map((x,y)->x.*y, sdual, ∆ldual)


Cu = create_curl(PRIM, g3.N, s∆ldual, ebc, e⁻ⁱᵏᴸ, reorder=true)
Cv = create_curl(DUAL, g3.N, s∆lprim, ebc, e⁻ⁱᵏᴸ, reorder=true)
CC = Cv * Cu


# Choose a guess solution.
Aₙₗ = spdiagm(param.εc) \ copy(CC)  # A for nonlasing mode
Ω², Ψ = eigs(Aₙₗ, nev=8, sigma=(1.5ωₐ)^2)
m = 3
ωₙₗ = √Ω²[m]
ψₙₗ = Ψ[:,m]
imax = indmax(abs, ψₙₗ)
aₙₗ = abs(ψₙₗ[imax])
ψₙₗ = ψₙₗ ./ ψₙₗ[imax]  # this does not only normalize, but changes phase, which is fine because we can freely change phase of each mode in SALT equation
# ψₙₗ = ψₙₗ ./ aₙₗ  # does not change phase, only normalize
nlsol = NonlasingSol([ωₙₗ], [ψₙₗ], [imax])
nlvar = NonlasingVar(CC, Mmax)

D₀_array = zeros(3, Nx)
dₗ₀ = 0.0
for dₙₗ = 0.1:0.05:0.35
    D₀_array .= 0
    D₀_array[:,1:100] .= dₙₗ
    param.D₀ .= D₀_array[:]
    for k = 1:20
        init_nlvar!(nlvar, 1, nlsol, param.D₀, CC, param)  # use param.D₀ because there is no lasing mode
        normnleq = norm_nleq(nlsol, 1, nlvar)
        # info("‖nleq‖ = $normnleq")
        if normnleq ≤ Base.rtoldefault(Float64)
            break
        end
        update_nlsol!(nlsol, 1, nlvar)
    end
    info("dₙₗ = $dₙₗ, ωₙₗ = $(nlsol.ω[1])")
    if imag(nlsol.ω[1]) > 0
        dₗ₀ = dₙₗ
        break
    end
end

# Let's check if the same sign mistake leads to the divergence in lasing case.

ωₗ = real(nlsol.ω[1])
ψₗ = nlsol.ψ[1]
imax = indmax(abs, ψₗ)
w = mod1(imax, 3)  # Cartesian component that is strongest
aₗ = abs(ψₗ[imax])
ψₗ = ψₗ ./ ψₗ[imax]  # this does not only normalize, but changes phase, which is fine because we can freely change phase of each mode in SALT equation
# ψₗ = ψₗ ./ aₗ  # does not change phase; only normalize
ψₗ₀ = copy(ψₗ)
aₗ = 0.0




# Lasing mode
# Create variables.
lsol = LasingSol([ωₗ], [aₗ], [ψₗ], [imax])
lvar = LasingVar(CC, Mmax)

# Procedure
# - Initialize ∆sol.  (This is a guess ∆sol to feed to the fixed-point equation solver.)
# - Initialize rvar and mvar for a given sol and ∆sol.
# - Feed the initialized ∆sol to update_∆sol.
# - Move sol by ∆sol.

for dₗ = dₗ₀:0.05:dₗ₀+1
    D₀_array .= 0
    D₀_array[:,1:100] .= dₗ
    param.D₀ .= D₀_array[:]
    normleq = 0.0
    for k = 1:100
        init_lvar!(lvar, lsol, CC, param)
        normleq = norm_leq(lsol, lvar)
        if normleq ≤ Base.rtoldefault(Float64)
            break
        end
        update_lsol!(lsol, lvar, CC, param)
    end
    info("‖leq‖ = $normleq")
    info("dₗ = $dₗ, ωₗ = $(lsol.ω[1]), aₗ = $(√lsol.a²[1])")
end

using PyPlot
# w=3
ψwₗ₀ = view(ψₗ₀, w:3:endof(ψₗ₀))
ψₗ = lsol.ψ[1]
ψwₗ = view(ψₗ, w:3:endof(ψₗ))
clf()
plot(1:Nx, abs.(ψwₗ₀), "ro", 1:Nx, abs.(ψwₗ), "b-")
# plot(1:Nx, abs.(ψwₗ₀), "ro", 1:Nx, abs.(lsol.a[1].*ψwₗ), "b-")
