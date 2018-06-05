export solve_leq!, solve_nleq!, pumpup!, simulate!

# solve_leq! is just a new name of anderson_salt!.
solve_leq!(lsol, lvar, CC, param; m=2, τr=1e-2, τa=1e-4, maxit=typemax(Int), verbose=true) =
    anderson_salt!(lsol, lvar, CC, param, m=m, τr=τr, τa=τa, maxit=maxit, verbose=verbose)

# Solve the nonlasing equations without the spatial hole-burning term.
function solve_nleq!(nlsol::NonlasingSol,
                     nlvar::NonlasingVar,
                     CC::AbsMatNumber,
                     param::SALTParam;
                     τ::Real=Base.rtoldefault(Float64),
                     maxit::Integer=20,  # maximum number of Newton iteration steps
                     verbose::Bool=true)
    mvec = 1:length(nlsol)  # 1:M
    D = param.D₀  # use param.D₀ because hole-burning is assumed zero
    for m = mvec
        # Newton method for nonlasing modes
        k = 0
        lnl = 0.0
        tic()
        for k = 1:maxit
            lnl = norm_nleq(m, nlsol, nlvar, D, CC, param)
            lnl ≤ τ && break
            update_nlsol!(nlsol, m, nlvar)
        end
        t_newton = toq()
        # verbose && println("\tmode $m: Newton steps = $k, ‖nleq‖ = $lnl")
        verbose && @printf("\tmode %d: Newton steps = %d (%f sec), ‖nleq‖ = %.3e\n", m, k, t_newton, lnl)
    end
end


# Solve the nonlasing equations with the spatial hole-burning term.
function solve_nleq!(nlsol::NonlasingSol,
                     nlvar::NonlasingVar,
                     lvar::LasingVar,  # used only to pass pre-calculated D
                     CC::AbsMatNumber,
                     param::SALTParam;
                     τ::Real=Base.rtoldefault(Float64),
                     maxit::Integer=20,  # maximum number of Newton iteration steps
                     verbose::Bool=true)
    # Below, lvar must be initialized already by init_lvar!, typically by norm_leq
    # within anderson_salt!
    mvec = nlsol.m_act
    D = lvar.rvar.D
    for m = mvec
        # Newton method for nonlasing modes
        k = 0
        lnl = 0.0
        tic()
        for k = 1:maxit
            lnl = norm_nleq(m, nlsol, nlvar, D, CC, param)
            lnl ≤ τ && break
            update_nlsol!(nlsol, m, nlvar)
        end
        t_newton = toq()
        # verbose && println("\tmode $m: Newton steps = $k, ‖nleq‖ = $lnl")
        verbose && @printf("\tmode %d: Newton steps = %d (%f sec), ‖nleq‖ = %.3e\n", m, k, t_newton, lnl)
    end
end


function pumpup!(lsol::LasingSol, lvar::LasingVar,
                 nlsol::NonlasingSol, nlvar::NonlasingVar,
                 CC::AbsMatNumber,
                 param::SALTParam,
                 dvec::AbsVecReal,  # trajectory of pump strength parameter to follow
                 setD₀!::Function;  # setD₀!(param, d) sets pump strength param.D₀ corresponding to pump strength parameter d
                 τ_newton::Real=Base.rtoldefault(Float64),
                 τr_anderson::Real=1e-2,  # relative tolerance; consider using Base.rtoldefault(Float)
                 τa_anderson::Real=1e-4,  # absolute tolerance
                 maxit_newton::Integer=20,  # maximum number of Newton iteration steps
                 maxit_anderson::Integer=typemax(Int),
                 verbose::Bool=true)
    println("\nPump up nonlasinge equations:")
    for d = dvec
        setD₀!(param, d)
        solve_nleq!(nlsol, nlvar, CC, param, τ=τ_newton, maxit=maxit_newton, verbose=verbose)
        verbose && println("d = $d, ω ₙₗ = $(string(nlsol.ω)[17:end])")  # 17 is to skip header "Complex{Float64}"
    end

    println("\nFind lasing modes at pumped point:")
    while turnon!(lsol, nlsol)
        tic()
        n_anderson, ll = solve_leq!(lsol, lvar, CC, param, τr=τr_anderson, τa=τa_anderson, maxit=maxit_anderson, verbose=verbose)
        t_anderson = toq()
        # verbose && println("\tAnderson steps = $k, ‖leq‖ = $ll, ω ₗ = $(lsol.ω), aₗ² = $(lsol.a²)")
        verbose && @printf("\tAnderson steps = %d (%f sec), ‖leq‖ = %.3e, ", n_anderson, t_anderson, ll); println("ω ₗ = $(lsol.ω), aₗ² = $(lsol.a²)")

        # Need to solve the nonlasing equation again.
        println("\tRecalculate nonlasing modes:")
        solve_nleq!(nlsol, nlvar, lvar, CC, param, τ=τ_newton, maxit=maxit_newton, verbose=verbose)
        verbose && println("\tω ₙₗ = $(string(nlsol.ω)[17:end])")  # 17 is to skip header "Complex{Float64}"
    end

    check_conflict(lsol, nlsol)
end


function simulate!(lsol::LasingSol, lvar::LasingVar,
                   nlsol::NonlasingSol, nlvar::NonlasingVar,
                   CC::AbsMatNumber,
                   param::SALTParam,
                   dvec::AbsVecReal,  # trajectory of pump strength parameter to follow
                   setD₀!::Function;  # setD₀!(param, d) sets pump strength param.D₀ corresponding to pump strength parameter d
                   outωaψ::NTuple{3,Bool}=(true,true,false),  # true to output ω, a, ψ
                   doutvec::AbsVecReal=dvec,  # output results when dvec[i] ∈ doutvec
                   τ_newton::Real=Base.rtoldefault(Float64),
                   τr_anderson::Real=1e-2,  # relative tolerance; consider using Base.rtoldefault(Float)
                   τa_anderson::Real=1e-4,  # absolute tolerance
                   maxit_newton::Integer=20,  # maximum number of Newton iteration steps
                   maxit_anderson::Integer=typemax(Int),
                   verbose::Bool=true)
    # Create output storages.
    outω, outa, outψ = outωaψ
    nout = length(doutvec)
    M = length(lsol)
    N = length(lsol.ψ[1])

    ωout = outω ? MatComplex(M,nout) : MatComplex(M,0)
    aout = outa ? zeros(M,nout) : zeros(M,0)  # use zeros because some entries of a²out won't be overwritten
    ψout = outψ ? Array{CFloat,3}(M,nout,N) : Array{CFloat,3}(M,0,N)
    nAA = zeros(Int, nout)  # nummber of Anderson acceleration steps
    tAA = zeros(nout)  # time taken for Anderson acceleration

    println("\nStart simulation.")
    cout = 1  # index of doutvec
    for d = dvec
        println("d = $d:")
        setD₀!(param, d)
        n_anderson = 0
        t_anderson = 0.0
        while true
            # Solve the lasing equations.
            while true
                tic()
                n_anderson, ll = solve_leq!(lsol, lvar, CC, param, τr=τr_anderson, τa=τa_anderson, maxit=maxit_anderson, verbose=false)
                t_anderson = toq()
                # verbose && println("\tAnderson steps = $k, ‖leq‖ = $ll, ω ₗ = $(lsol.ω), aₗ² = $(lsol.a²)")
                verbose && @printf("\tAnderson steps = %d (%f sec), ‖leq‖ = %.3e, ", n_anderson, t_anderson, ll); println("ω ₗ = $(lsol.ω), aₗ² = $(lsol.a²)")
                if !shutdown!(lsol, nlsol)
                    break
                end
            end

            # Solve the nonlasing equation.
            println("\tRecalculate nonlasing modes:")
            solve_nleq!(nlsol, nlvar, lvar, CC, param, τ=τ_newton, maxit=maxit_newton, verbose=true)
            verbose && println("\tω ₙₗ = $(string(nlsol.ω)[17:end])")  # 17 is to skip header "Complex{Float64}"
            if !turnon!(lsol, nlsol)
                break
            end
        end

        check_conflict(lsol, nlsol)

        # Output results for current d.
        if d == doutvec[cout]
            nAA[cout] = n_anderson
            tAA[cout] = t_anderson

            # Output the lasing modes.
            m_act = lsol.m_act
            outω && (ωout[m_act,cout] .= lsol.ω[m_act])
            outa && (aout[m_act,cout] .= .√lsol.a²[m_act])
            if outψ
                for m = m_act
                    ψout[m,cout,:] .= lsol.ψ[m]
                end
            end

            # Output nonlasing modes.
            m_act = nlsol.m_act  # this does not conflict with lsol.m_act, because check_conflict() passed
            outω && (ωout[m_act,cout] .= nlsol.ω[m_act])
            if outψ
                for m = m_act
                    ψout[m,cout,:] .= nlsol.ψ[m]
                end
            end

            cout += 1
        end
    end

    return ωout, aout, ψout, nAA, tAA
end
