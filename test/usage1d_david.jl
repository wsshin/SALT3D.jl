using SALT3D

Ngrid = 30
Lcavity = 1.
hx = Lcavity / (Ngrid + 1)

ωn(n) = sqrt( -2/hx^2 * (cos(n*pi/(Ngrid+1))-1) )
nmode = 2
assert(0 < nmode <= Ngrid)
x = linspace(hx, Lcavity-hx, Ngrid)
ψn(n) = sin.(n*pi*x/Lcavity)

ωₐ = ωn(nmode)  # how do we choose ωₐ in general
γ⟂ = 1
γ(ω) = γ⟂ / (ω - ωₐ + 1im*γ⟂)
∂γ∂ω(ω) = -γ(ω)^2 / γ⟂
σ = 0.03
ɛc = 1 + im * σ

ɛ = ɛc * ones(Ngrid)
diags = (1/hx^2) * -2 * ones(Ngrid)
off_diags = (1/hx^2) * 1 * ones(Ngrid-1)
∇² = full( SymTridiagonal( diags, off_diags ) );

ωguess = ωₐ
ψguess = ψn(nmode)  # so, David used nonlasing mode for d = 0 for guess for lasing mode
Dpump = σ*1.02  # slightly higher than loss for lasing; increase this from 1.02 to 10.0 and plot 20*imag.(ψ) to see difference.
cguess = 1.0;
tol = 1e-7

imax = indmax(abs.(ψguess))
ψ = ψguess / ψguess[imax]
ω = ωguess
c = cguess

block1, block2 = 1:Ngrid, Ngrid+1:2Ngrid
row1, row2 = 2Ngrid+1, 2Ngrid+2
column1, column2 = 2Ngrid+1, 2Ngrid+2

its = 0
while true
    # everything here is a vector except Mmatrix
    H = 1 ./ (1+c^2*abs.(ψ).^2)
    Mmatrix = ∇² + ω^2 * diagm(ɛ + Dpump*γ(ω)*H)

    f = zeros(2Ngrid+2)
    v = zeros(2Ngrid+2)
    f[block1] = real(Mmatrix * ψ)
    f[block2] = imag(Mmatrix * ψ)
    f[row1] = real(ψ[imax]) - 1
    f[row2] = imag(ψ[imax])

    normf = norm(f)
    @printf("its = %i, c = %g, |f| = %e\n", its, c, normf)
    if normf < tol
        break
    else
        its +=1
    end

    v[block1] = real(ψ)
    v[block2] = imag(ψ)
    v[row1] = ω
    v[row2] = c

    ∂H∂c = -2c * abs.(ψ).^2 .* H.^2
    ∂H∂ψR = -2c^2 * H.^2 .* real(ψ)
    ∂H∂ψI = -2c^2 * H.^2 .* imag(ψ)
    ∂M∂ω = 2ω * (ɛ + Dpump*γ(ω)*H) + ω^2 * Dpump*∂γ∂ω(ω)*H
    ∂M∂c = ω^2 * Dpump * γ(ω) * ∂H∂c
    ∂M∂ψRR = real( ω^2 * Dpump * γ(ω) * ∂H∂ψR .* ψ )
    ∂M∂ψRI = real( ω^2 * Dpump * γ(ω) * ∂H∂ψI .* ψ )
    ∂M∂ψIR = imag( ω^2 * Dpump * γ(ω) * ∂H∂ψR .* ψ )
    ∂M∂ψII = imag( ω^2 * Dpump * γ(ω) * ∂H∂ψI .* ψ)

    J = zeros(2Ngrid+2, 2Ngrid+2)
    J[block1, block1] = real(Mmatrix) + diagm(∂M∂ψRR)
    J[block1, block2] = -imag(Mmatrix) + diagm(∂M∂ψRI)
    J[block2, block1] = imag(Mmatrix) + diagm(∂M∂ψIR)
    J[block2, block2] = real(Mmatrix) + diagm(∂M∂ψII)
    J[block1, column1] = real(∂M∂ω .* ψ)
    J[block2, column1] = imag(∂M∂ω .* ψ)
    J[block1, column2] = real(∂M∂c .* ψ)
    J[block2, column2] = imag(∂M∂c .* ψ)
    J[row1, imax] = 1
    J[row2, Ngrid+imax] = 1

    dv = -J \ f
    ψ += dv[block1] + im * dv[block2]
    ω += dv[row1]
    c += dv[row2]

end

using PyPlot

plot(1:Ngrid, ψguess, 1:Ngrid, 20.*imag.(ψ))
