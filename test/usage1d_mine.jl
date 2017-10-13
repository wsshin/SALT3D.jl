using SALT3D

# Set up a system.
Ngrid = 100
Lcavity = 1.
hx = Lcavity / (Ngrid + 1)

ωn(n) = (2/hx) * sin(n*π*hx / (2*Lcavity))
nmode = 2
assert(0 < nmode <= Ngrid)  # The discretized Laplacian is Ngrid × Ngrid, so the system has Ngrid eigenvalues
x = linspace(hx, Lcavity-hx, Ngrid)
ψn(n) = sin.(n*pi*x/Lcavity)

ωₐ = ωn(nmode)
γ⟂ = 1
γ(ω) = γ⟂ / (ω - ωₐ + im*γ⟂)
∂γ∂ω(ω) = -γ(ω)^2 / γ⟂
σ = 0.03
εc = 1 + im * σ

ε = εc * ones(Ngrid)
diags = (1/hx^2) * -2 * ones(Ngrid)
off_diags = (1/hx^2) * 1 * ones(Ngrid-1)
∇² = full( SymTridiagonal( diags, off_diags ) );

# Set up basic parameters and functions for the solver.
ωguess = ωₐ
ψguess = ψn(nmode)
D0 = σ*1.02
cguess = 1.0;
tol = 1e-7

imax = indmax(abs.(ψguess))
ψguess = ψguess / ψguess[imax]  # makes ψ[imax] = 1
v_vec = [ψguess; ωguess + im*cguess]

D(ψ, c) = D0 ./ (1+c^2*abs.(ψ).^2)
∂D∂a(ψ, c) = -D0 ./ (1+c^2*abs.(ψ).^2).^2  # differentiation in a = (denominator)

compfy(x) = x[1:length(x)÷2] + im * x[length(x)÷2+1:end]
realfy(x) = [real(x); imag(x)]

# Set up functions for implicit Newton step calculation.

# SALT operator
Msalt(ψ,ω,c) = ∇² + ω^2 * diagm(ε + γ(ω)*D(ψ,c))
NLterm(ψ,∆ψ,c) = ∂D∂a(ψ,c) .* ψ .* real(conj(ψ) .* ∆ψ)

# SALT equation; v satisfying f(v) = 0 is the solution we want.
function f(v)
    ψ, ω, c = v[1:Ngrid], real(v[Ngrid+1]), imag(v[Ngrid+1])

    return Msalt(ψ,ω,c) * ψ
end

function invA_term1(v, ∆v)
    ψ, ω, c = v[1:Ngrid], real(v[Ngrid+1]), imag(v[Ngrid+1])
    ∆ψ = ∆v[1:Ngrid]
    ω², c² = ω^2, c^2

    return Msalt(ψ,ω,c) \ (f(v) + 2ω²*γ(ω)*c² * NLterm(ψ,∆ψ,c))  # for ∆ψ = 0, NLterm = 0, so it becomes M \ f(v)
end

function invA_∂f∂ω(v, ∆v)
    ψ, ω, c = v[1:Ngrid], real(v[Ngrid+1]), imag(v[Ngrid+1])
#    ∆ω = real(∆v[Ngrid+1])
    ω² = ω^2

    ∂M∂ω = 2ω * (ε + γ(ω)*D(ψ,c)) + ω²*∂γ∂ω(ω) * D(ψ,c)

    return Msalt(ψ,ω,c) \ (∂M∂ω .* ψ)
end

function invA_∂f∂c(v, ∆v)
    ψ, ω, c = v[1:Ngrid], real(v[Ngrid+1]), imag(v[Ngrid+1])
#    ∆c = imag(∆v[Ngrid+1])
    ω² = ω^2

    ∂M∂c = 2c*ω²*γ(ω) * (abs.(ψ).^2.* ∂D∂a(ψ,c))

#    println("∂M∂c = $∂M∂c\n")
#    println("c = $c")
#    println("ω = $ω")
#    println("ψ = $ψ")
#    println("∂D∂a = $(∂D∂a(ψ,c))")

    return Msalt(ψ,ω,c) \ (∂M∂c .* ψ)
end

function ∆v_new(v, ∆v)
    println("norm(∆v) = $(norm(∆v))")
    ∆ψ_term1 = invA_term1(v, ∆v)
    ∆ψ_ωvec = invA_∂f∂ω(v, ∆v)
    ∆ψ_cvec = invA_∂f∂c(v, ∆v)

    println("norm(∆ψ_term1) = $(norm(∆ψ_term1)), norm(∆ψ_ωvec) = $(norm(∆ψ_ωvec)), norm(∆ψ_cvec) = $(norm(∆ψ_cvec))")

#    println("∆ψ_cvec = $∆ψ_cvec")

    z1 = ∆ψ_ωvec[imax]
    z2 = ∆ψ_cvec[imax]
    w = -∆ψ_term1[imax]

    M = [realfy(z1) realfy(z2)]
    b = realfy(w)

    println("M = $M")

    ∆q = compfy(M\b)
    ∆ω = real(∆q)
    ∆c = imag(∆q)

    ∆ψ = -∆ψ_term1 - ∆ψ_ωvec .* ∆ω - ∆ψ_cvec .* ∆c

    return [∆ψ; ∆q]
end

p(f) = x -> let y=f(x)
    global num_step
    println("Step $(num_step += 1) of Anderson acceleration")
    y
end

# num_step = 0
# x = andersonaccel(p(∆v -> ∆v_new(v_vec, ∆v)), complex(zeros(Ngrid+1)), m=1, maxit=Ngrid);


### g_combined ###
# function g_combined(v∆v)
#     v∆v_old = v∆v
#     v_vec = v∆v[1:Ngrid+1]
#     ∆v = v∆v[Ngrid+2:end]
#
# #    ∆v = andersonaccel(∆v -> realfy(∆v_new(v_vec, compfy(∆v))), zeros(2Ngrid+4), m = 3, maxit = Ngrid, reltol=1e-3)
# #    ∆v = compfy(∆v)
#     ∆v = ∆v_new(v_vec, ∆v)
# #    println("∆v = $∆v")
#
#     v_vec += ∆v
#
#     v∆v_new = [v_vec; ∆v]
#
#     norm_v∆v = norm(v∆v_new - v∆v_old)
# #    println("norm_v∆v = $norm_v∆v")
#
#     return v∆v_new
# end
#
# num_step = 0
#
# v_vec_guess = complex([ψguess; ωguess + im*cguess])
# #v_vec_guess = [rand(Ngrid)+im*rand(Ngrid); rand() + im*rand()]
# #v_vec_guess = complex([ψguess; rand() + im*cguess])
# #v_vec_guess = complex([ψguess; ωguess*1.05 + im*cguess])
# v_vec_guess[imax] = 1.0
#
# #     dv = andersonaccel(p(∆v -> realfy(∆v_new(v_vec_guess, compfy(∆v)))), zeros(2Ngrid+4), m = 3, maxit = Ngrid, reltol = 1e-3)
# #∆v_guess = andersonaccel(p(∆v -> realfy(∆v_new(v_vec_guess, compfy(∆v)))), zeros(2Ngrid+2), m = 3, maxit = Ngrid)
# #∆v_guess = compfy(∆v_guess)
# ∆v_guess = complex(zeros(Ngrid+1))
#
# println("\nInitial guess ∆v created.\n")
#
# v∆v_guess_re = realfy([v_vec_guess; ∆v_guess])
#
# v∆v_sol_re = andersonaccel(p(v∆v_re -> realfy(g_combined(compfy(v∆v_re)))), v∆v_guess_re, m = 0, maxit = Ngrid)
#
# v∆v_sol = compfy(v∆v_sol_re)
# v_vec = v∆v_sol[1:Ngrid+1]
#
# f_vec = f(v_vec)
# normf = norm(f_vec)
#
# ψ = v_vec[1:Ngrid]
# ω = real(v_vec[Ngrid+1])
# c = imag(v_vec[Ngrid+1])
#
# println("ω = $ω, c = $c, |f| = $normf");

### g_combined0
function g_combined0(v_vec)
    ψ, ω, c = v_vec[1:Ngrid], real(v_vec[Ngrid+1]), imag(v_vec[Ngrid+1])

#    @printf("its = %i, ω = %f, c = %f, |f| = %e\n", its, real(ω), real(c), normf)
#    println("its = $its, ω = $ω, c = $c")

#    ∆v = andersonaccel(∆v -> realfy(∆v_new(v_vec, compfy(∆v))), zeros(2Ngrid+4), m = 3, maxit = Ngrid, reltol=1e-7)
#    ∆v = compfy(∆v)
#    ∆v = ∆v_new(v_vec, ∆v)
    ∆v = ∆v_new(v_vec, zeros(Ngrid+1))
#    println("∆v = $∆v")

    v_vec += ∆v

    return v_vec
end

num_step = 0

v_vec_guess = complex([ψguess; ωguess + im*cguess])
#v_vec_guess = [rand(Ngrid)+im*rand(Ngrid); rand() + im*rand()]
#v_vec = complex([ψguess; rand() + im*rand()])
#v_vec_guess = complex([ψguess; rand() + im*cguess])
v_vec_guess[imax] = 1.0

# Set m = 1 to compare with Newton's method.
# When rtol is not set sufficiently large, the solution diverge!
# v_vec_sol_re = andersonaccel(p(v_vec_re -> realfy(g_combined0(compfy(v_vec_re)))), realfy(v_vec_guess), m = 0, maxit = Ngrid)
# v_vec = compfy(v_vec_sol_re)

# Complex version
v_vec = andersonaccel(p(v_vec_re -> g_combined0(v_vec_re)), v_vec_guess, m = 0, maxit = Ngrid)


f_vec = f(v_vec)
normf = norm(f_vec)

ψ, ω, c = v_vec[1:Ngrid], real(v_vec[Ngrid+1]), imag(v_vec[Ngrid+1])

#@printf("ω = %f, c = %f, |f| = %e\n", real(ω), real(c), normf)
println("ω = $ω, c = $c, |f| = $normf");


using PyPlot
plot(1:Ngrid, ψguess, 1:Ngrid, abs.(v_vec[1:Ngrid]))
println("c = $c")  # must be around 0.163459
