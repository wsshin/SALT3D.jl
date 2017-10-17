using SALT3D
using MaxwellFD3D
using JLD
using PyPlot

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


# Choose guess solutions
# Aₙₗ = spdiagm(param.εc) \ copy(CC)  # A for nonlasing mode
# Ω², Ψ = eigs(Aₙₗ, nev=8, sigma=(1.3ωₐ)^2)  # 1.3ωₐ gives ω ≈ 70, 100, 130, 160
# @save "guess_multimode.jld" Ω² Ψ
@load "guess_multimode.jld"

ms = 1:2:8
M = length(ms)  # number of modes of interest
ωₙₗ = .√Ω²[ms]
Ψₙₗ = Ψ[:,ms]
nlsol = NonlasingSol(ωₙₗ, Ψₙₗ)
normalize!(nlsol)
ψₙₗ₀ = deepcopy(nlsol.ψ)

nlvar = NonlasingVar(CC, M)

lsol = LasingSol(3Nx, M)
lvar = LasingVar(CC, M)

D₀_array = zeros(3, Nx)
dₙₗₛ = 0.0
dₙₗₑ = 0.45
# dₙₗₑ = 0.0

info("Pump up to the desired starting point.")
for dₙₗ = dₙₗₛ+0.05:0.01:dₙₗₑ
    D₀_array .= 0
    D₀_array[:,1:100] .= dₙₗ
    param.D₀ .= D₀_array[:]
    for m = 1:length(nlsol)
        # Newton method for nonlasing modes
        for k = 1:20
            lnl = norm_nleq(m, nlsol, nlvar, param.D₀, CC, param)  # use param.D₀ because hole-burning is assumed zero
            info("m = $m, ‖nleq‖ = $lnl")
            lnl ≤ Base.rtoldefault(Float64) && break
            update_nlsol!(nlsol, m, nlvar)
        end
    end
    info("dₙₗ = $dₙₗ, ωₙₗ = $(nlsol.ω)")
end

println()
info("Find the initial lasing modes.")
while turnon!(lsol, nlsol)
    # info("nlsol.ω[1] = $(nlsol.ω[1])")
    anderson_salt!(lsol, lvar, CC, param)
    println("ωₗ = $(lsol.ω), aₗ² = $(lsol.a²)")

    # Need to solve the nonlasing equation again.
    for m = nlsol.m_act
        # Newton method for nonlasing modes
        for k = 1:20
            lnl = norm_nleq(m, nlsol, nlvar, lvar.rvar.D, CC, param)
            info("m = $m, ‖nleq‖ = $lnl")
            lnl ≤ Base.rtoldefault(Float64) && break
            update_nlsol!(nlsol, m, nlvar)
        end
    end
end
info("Result:")
println("ωₗ = $(lsol.ω), aₗ² = $(lsol.a²)")

lsol₀ = deepcopy(lsol)

check_conflict(lsol, nlsol)

println()
info("Start simulation.")
for dₗ = dₙₗₑ+0.5:0.5:dₙₗₑ+3  # too many fixed-point iterations if last value is too large, even if M = 2
    D₀_array .= 0
    D₀_array[:,1:100] .= dₗ
    param.D₀ .= D₀_array[:]
    while true
        # Solve the lasing equation.
        while true
            anderson_salt!(lsol, lvar, CC, param)
            println("dₗ = $dₗ, ωₗ = $(lsol.ω), aₗ² = $(lsol.a²)")
            if !shutdown!(lsol, nlsol)
                break
            end
        end

        # Solve the nonlasing equation.
        for m = nlsol.m_act
            # Newton method for nonlasing modes
            for k = 1:20
                lnl = norm_nleq(m, nlsol, nlvar, lvar.rvar.D, CC, param)
                println("m = $m, ‖nleq‖ = $lnl")
                lnl ≤ Base.rtoldefault(Float64) && break
                update_nlsol!(nlsol, m, nlvar)
            end
        end
        println("dₗ = $dₗ, ωₙₗ = $(nlsol.ω)")
        if !turnon!(lsol, nlsol)
            break
        end
    end
    check_conflict(lsol, nlsol)
end


using PyPlot
m = 1
# assert(nlsol.iₐ[m]==lsol₀.iₐ[m]==lsol.iₐ[m])  # error if dₙₗₑ = 0.0 and lsol₀ is uninitialized.
assert(nlsol.iₐ[m]==lsol.iₐ[m])  # error if dₙₗₑ = 0.0 and lsol₀ is uninitialized.
w = mod1(nlsol.iₐ[m], 3)  # Cartesian component that is strongest

ψₙₗ₀ₘ = ψₙₗ₀[m]
ψwₙₗ₀ₘ = view(ψₙₗ₀ₘ, w:3:endof(ψₙₗ₀ₘ))

ψₗ₀ₘ = lsol₀.ψ[m]
ψwₗ₀ₘ = view(ψₗ₀ₘ, w:3:endof(ψₗ₀ₘ))

ψₗₘ = lsol.ψ[m]
ψwₗₘ = view(ψₗₘ, w:3:endof(ψₗₘ))

clf()
plot(1:Nx, abs.(ψwₙₗ₀ₘ), "r-.", 1:Nx, abs.(ψwₗ₀ₘ), "g--", 1:Nx, abs.(ψwₗₘ), "b-")
