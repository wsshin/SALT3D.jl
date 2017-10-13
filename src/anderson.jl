export andensoraccel!, andersonaccel

# To do: pass the storages f, ∆X, ∆F, Q, γ.  Also, think about ways to reduce this memory.

# Solve the fixed-point equation g(x) = x to given relative and absolute tolerances,
# returning the solution x, via Anderson acceleration of a fixed-point iteration starting at
# a given initial x (which must be of the correct type to hold the result).
#
# This implementation operates in-place as much as possible.
function andersonaccel!(
    g!::Function,  # g!(y,x) that overwrites a given output vector y with g(x); x and y must be different vectors
    x::AbstractVector{T};  # initial guess; overwritten at each iteration step (note g! itself does not overwrite x)
    m = min(10, length(x)÷2), # number of additional x's kept in algorithm; m = 0 means unaccelerated iteration
    norm=Base.norm, # stopping criterion: norm(Δx) ≤ rtol*norm(x) + atol
    rtol::Real=Base.rtoldefault(real(T)),  # relative tolerance (real(T) = R)
    atol::Real=0,  # absolute tolerance
    maxit::Int=typemax(Int)  # number of maximum number of iteration steps
) where {R<:AbstractFloat,T<:Union{R,Complex{R}}}

    m ≥ 0 || throw(ArgumentError("m (= $m) < 0 is not allowed."))
    rtol ≥ 0 || throw(ArgumentError("rtol (= $rtol) < 0 is not allowed."))
    atol ≥ 0 || throw(ArgumentError("atol (= $atol) < 0 is not allowed."))

    n = length(x)
    m ≤ n || throw(ArgumentError("m (= $m) > n (= $n) is not allowed."))

    y = Vector{T}(n)

    if m == 0 # simple fixed-point iteration, no memory
        for k = 0:maxit-1
            g!(y, x)

            # Use y to store ∆x temporarily; y will be updated to y = g(x) in the next
            # iteration step anyway.
            for i = eachindex(x)
                x[i], y[i] = y[i], y[i]-x[i]
            end

            lx, ly = norm(x), norm(y)
            info("Step $(k+1) of Anderson: ‖x($(k+1))-x($k)‖ / ‖x($k)‖ = $(ly/lx)")
            ly ≤ rtol*lx + atol && break  # norm(y) = norm(∆x)
        end
    else  # m ≠ 0
        # Pre-allocate all of the arrays we will need.  The goal is to allocate once and re-use
        # the storage during the iteration by operating in-place.
        f = Vector{T}(n)
        ∆X = Matrix{T}(n, m)  # columns: ∆x's
        ∆F = Matrix{T}(n, m)  # columns: ∆f's
        Q = Matrix{T}(n, m)  # space for QR factorization
        γ = Vector{T}(max(n,m))  # not m, in order to store RHS vector (f: length-n) and overwrite in-place via A_ldiv_B! (max length m)

        # First iteration: update y = g(x).
        g!(y, x)

        kcol = 1
        for i = eachindex(x)
            x[i], ∆X[i,kcol] = y[i], (y[i] = f[i] = y[i]-x[i])  # f(x) = g(x) - x
        end

        # Subsequent iterations
        for k = 1:maxit-1
            norm(y) ≤ rtol*norm(x) + atol && break

            # Update y = g(x).
            g!(y, x)

            # Prepare the least squares problem.
            for i = eachindex(x)
                f[i], ∆F[i,kcol] = (γ[i] = y[i]-x[i]), γ[i]-f[i]  # γ = fnew
            end

            m′ = min(m, k)

            # Construct subarrays to work in-place on a subset of the columns.
            if m′ < m
                γ′, ∆X′, ∆F′, Q′ = @view(γ[1:m′]), @view(∆X[:,1:m′]), @view(∆F[:,1:m′]), @view(Q[:,1:m′])
                # γ′, ∆X′, ∆F′, Q′ = @views γ[1:m′], ∆X[:,1:m′], ∆F[:,1:m′], Q[:,1:m′]  # in Julia 0.7
            elseif m′ == m
                γ′, ∆X′, ∆F′, Q′ = @view(γ[1:m]), ∆X, ∆F, Q
            end  # use previous γ′, ∆X′, ∆F′, Q′ for m′ > m

            # Solve the least squares problem.
            QR = qrfact!(copy!(Q′,∆F′), Val{true})
            A_ldiv_B!(QR, γ) # overwrites γ′ in-place with ∆F′ \ f; only first m′ entries of γ are meaningful

            # Replace columns of ∆F and ∆X with the new data in-place.  Rather than always appending
            # the new data in the last column, we cycle through the m-1 columns periodically.
            kcol = mod1(kcol+1, m)  # next column of ∆F, ∆X to update

            # x = x + f - (∆X′ + ∆F′)*γ′, updating in-place
            # (also update ∆X[:,kcol+1] with new Δx, and set y = Δx)
            for i = eachindex(x)
                y[i] = f[i]
                for j = eachindex(γ′)
                    y[i] -= (∆X′[i,j] + ∆F′[i,j])*γ′[j]
                end

                # At this point, y is temporary storage for ∆x.
                ∆X[i,kcol] = y[i]
                x[i] += y[i]
            end
        end  # for k = 1:maxit-1
    end  # if m == 0

    return x
end

"""
    andersonaccel(g, x; m, norm=Base.norm, rtol=sqrt(ɛ), atol=0, maxits)

Solve the fixed-point problem \$g(x)=x\$ to given relative and absolute tolerances,
returning the solution \$x\$, via Anderson acceleration of a fixed-point
iteration starting at `x` and given the function `g(x)`.

The keyword parameters are the same as for `andersonaccel`: they specify
the "memory" `m` of the algorithm, the relative (`rtol`) and absolute
(`atol`) stopping tolerances in the given `norm`, and the maximum
number of itertions (`maxits`).
"""
andersonaccel{T<:Number}(g, x::AbstractVector{T}; kws...) = andersonaccel!(
    (y,x) -> copy!(y, g(x)),  # convert y = g(x) to g!(y,x) that overwrites y
    copy!(Array{float(T)}(length(x)), x);  # to prevent in-place solution
    kws...)

andersonaccel{T<:Number}(g, x::T; kws...) = andersonaccel(
    x -> g(x[1]),  # g takes scalar
    [x];
    kws...)[1]
