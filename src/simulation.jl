export pump!, simulate!, find_threshold!

const TA_NEWTON = Base.rtoldefault(Float)  # absolute tolerance for Newton method to solve nonlasing equation
const TR_NEWTON = sqrt(TA_NEWTON)  # relative tolerance for Newton method to solve nonlasing equation
const MAXIT_NEWTON = 20  # maximum number of Newton iteration steps

const TA_ANDERSON = Base.rtoldefault(Float) # absolute tolerance for Anderson acceleration to solve lasing equation
const TR_ANDERSON = sqrt(TA_ANDERSON)  # relative tolerance for Anderson acceleration to solve lasing equation; consider using Base.rtoldefault(Float)
const MAXIT_ANDERSON = 20  # maximum number of Anderson iteration steps
const M_ANDERSON = 2  # number of basis vectors to use in Anderson acceleration

const TA_THRESHOLD = eps(Float)  # relative tolerance of bisection method
const TR_THRESHOLD = sqrt(TA_THRESHOLD)  # relative tolerance of bisection method
const MAXIT_THRESHOLD = 50  # maximum number of bisection steps

# solve_leq! is just a new name of anderson_salt!.  init_lval!, norm_leq, and update_lsol!
# are called inside anderson_salt!.
function solve_leq!(lsol, lvar, param; m=M_ANDERSON, τr=TR_ANDERSON, τa=TA_ANDERSON, maxit=MAXIT_ANDERSON, verbose=true)
    println("  Solve lasing eq. with modes m = $(lsol.m_active) where aₗ²[m=1:$(length(lsol))] = $(lsol.a²):")

    t_anderson = @elapsed n_anderson, ll, ll₀ = anderson_salt!(lsol, lvar, param, m=m, τr=τr, τa=τa, maxit=maxit, verbose=verbose)

    # verbose && println("  No. of Anderson steps = $k, ‖leq‖ = $ll, ωₗ = $(lsol.ω), aₗ² = $(lsol.a²)")
    verbose && @printf("    No. of Anderson steps = %d (%f sec), ‖leq‖ = %.3e, ‖leq‖/‖leq₀‖ = %.3e\n", n_anderson, t_anderson, ll, ll/ll₀)
    verbose && println("    Lasing solution: aₗ² = $(lsol.a²), ωₗ = $(lsol.ω)")

    return n_anderson, t_anderson
end

# Solve the nonlasing equation without the spatial hole-burning term.
#
# The default value of D = param.D₀ is when the hole-burning term is zero (i.e., when D = D₀).
# Otherwise, pass LasingVar.LasingReducedVar.D for the argument D.  LasingVar must be
# initialized by init_lvar! in order to hold a properly calculated popolation inversion D.
# This is typically done by norm_leq, which is typically executed by calling anderson_salt!,
# which is called solve_leq! in the present context.
function solve_nleq!(nlsol::NonlasingSol,
                     nlvar::NonlasingVar,
                     param::SALTParam,
                     D::AbsVecFloat=param.D₀;
                     τr::Real=TR_NEWTON,  # relative tolerance
                     τa::Real=TA_NEWTON,  # absolute tolerance
                     maxit::Integer=MAXIT_NEWTON,  # maximum number of Newton iteration steps
                     verbose::Bool=true)
    τr ≥ 0 || throw(ArgumentError("τr = $τr must be ≥ 0."))
    τa ≥ 0 || throw(ArgumentError("τa = $τa must be ≥ 0."))
    maxit ≥ 0 || throw(ArgumentError("maxit = $maxit must be ≥ 0."))

    println("  Solve nonlasing eq. with modes m = $(nlsol.m_active) where ωₙₗ[m=1:$(length(nlsol))] = $(string(nlsol.ω)[17:end]):")

    for m = nlsol.m_active
        # Newton method for nonlasing modes
        t_newton = @elapsed begin
            k = 0
            init_nlvar!(nlvar, m, nlsol, D, param)
            lnleq₀ = norm_nleq(m, nlsol, nlvar)
            # verbose && println("  Initial residual norm: ‖nleq₀‖ = $lnleq₀")

            τ = max(τr*lnleq₀, τa)
            lnleq = lnleq₀
            while (k+=1) ≤ maxit
                lnleq ≤ τ && break
                update_nlsol!(nlsol, m, nlvar)
                init_nlvar!(nlvar, m, nlsol, D, param)
                lnleq = norm_nleq(m, nlsol, nlvar)
            end
        end
        # verbose && println("  mode $m: No. of Newton steps = $k, ‖nleq‖ = $lnleq")
        verbose && @printf("    mode %d: No. of Newton steps = %d (%f sec), ‖nleq‖ = %.3e, ‖nleq‖/‖nleq₀‖ = %.3e\n", m, k, t_newton, lnleq, lnleq/lnleq₀)
    end

    verbose && println("    Nonlasing solution: ωₙₗ = $(string(nlsol.ω)[17:end])")  # 17 is to skip header "Complex{Float64}"
end


# Solve the SALT lasing and nonlasing equations for a given pump parameter.
#
# In the original pump up algorithm described in Esterhazy et al.'s paper, the nonlasing
# equation is kept being solved while the system is being pumped up, and the lasing equation
# is solved once the target pump parameter is reached, by turning on modes one by one
# starting from the nonlasing mode with the largest positive Im{ω}.  Once the turned-on
# lasing mode is computed, the nonlasing modes are recalculated with the lasing modes in the
# hole-burning term to see if all the remaining nonlasing modes have negative Im{ω}.  If
# some nonlasing modes still have positive Im{ω}, the one with the largest positive Im{ω} is
# turned on again, and the procedure is repeated until all the remaining nonlasing modes have
# nonpositve Im{ω}.
#
# From this pump-up procedure, for every pump parameter in simulate! we are tempted to solve
# the nonlasing equation first, pick the mode to turn on, and then solve the lasing equation.
#
# However, this turns out to be a bad idea.  In fact, solving the lasing equation first at
# every pump parameter is a better.  If we solve the nonlasing equation first with the new
# pump parameter, we need to construct the nonlasing equation with the lasing modes for the
# old pump parameter.  If some nonlaisng modes described by this equation have positive
# Im{ω}, we would turn on one of them, and solve the lasing equation including the newly
# turned-on mode.  However, this newly turned-on mode could have stayed nonlasing if the
# lasing equation had been solved first, because then the updated lasing modes in the hole-
# burning term would have reduced Im{ω}'s of the nonlasing modes.  So, solving the nonlasing
# equation first for the new pump parameter has a possibility of unnecessary turning-on.
#
# Therefore, we write a function that calculate modes for a new pump parameter by solving
# the lasing equation first and then nonlasing equaiton.  Assume lsol and nlsol contain
# good guess solutions.
function solve_salt!(lsol::LasingSol, lvar::LasingVar,
                     nlsol::NonlasingSol, nlvar::NonlasingVar,
                     param::SALTParam,
                     d::Real,  # pump strength parameter to calculate the modes for
                     setD₀!::Function;  # setD₀!(param, d) sets pump strength param.D₀ corresponding to pump strength parameter d
                     τr_newton::Real=TR_NEWTON,  # relative tolerance for Newton method to solve nonlasing equation
                     τa_newton::Real=TA_NEWTON,  # absolute tolerance for Newton method to solve nonlasing equation
                     maxit_newton::Integer=MAXIT_NEWTON,  # maximum number of Newton iteration steps
                     m_anderson::Integer=M_ANDERSON,  # number of basis vectors to use in Anderson acceleration
                     τr_anderson::Real=TR_ANDERSON,  # relative tolerance for Anderson acceleration to solve lasing equation; consider using Base.rtoldefault(Float)
                     τa_anderson::Real=TA_ANDERSON,  # absolute tolerance for Anderson acceleration to solve lasing equation
                     maxit_anderson::Integer=MAXIT_ANDERSON,  # maximum number of Anderson iteration steps
                     verbose::Bool=true)
    println("d = $d:")
    setD₀!(param, d)
    n_anderson = 0
    t_anderson = 0.0
    lsol.activated .= false  # no lasing mode is just activated
    nlsol.activated .= false  # no nonlasing mode is just activated
    while true
        # Solve the lasing equation.
        # Calculating the lasing modes first is doable because the lasing equation can
        # be constructed for lasing modes without knowing nonlasing modes.
        while true
            # Below, solve_leq! takes the indices of the modes that lased for the
            # previous D₀ and updates the corresponding modes by solving the lasing
            # equation for the new D₀.
            n_anderson, t_anderson = solve_leq!(lsol, lvar, param, τr=τr_anderson, τa=τa_anderson, maxit=maxit_anderson, m=m_anderson, verbose=verbose)

            # Below, shutdown! checks if some of the newly calculated lasing modes have
            # a² ≤ 0.  If some do, it picks the one with the most negative a² and shuts
            # it down.
            #
            # Even if multiple lasing modes had a² ≤ 0, shutting down one mode releases the
            # power it held to all the other lasing modes and therefore could make all the
            # remaining nonpositive a² positive again.  Therefore, we must not shut down all
            # the lasing modes with a² ≤ 0, but do so only one-by-one.
            m_shutdown = shutdown!(lsol, nlsol)

            # If some mode was shut down above, continue the loop to perform solve_leq!
            # again to recalculate the lasing modes only with the indices of the
            # still-lasing modes.
            #
            # If no mode was shut down, break the loop.
            m_shutdown ≠ 0 || break
        end

        # Solve the nonlasing equation.
        # Below, solve_nleq! constructs the nonlasing equation for the new D₀ with the
        # already-calculated lasing modes and updates the nonlasing modes by solving the
        # equation.
        @assert lvar.inited  # lvar.rvar.D is initialized
        solve_nleq!(nlsol, nlvar, param, lvar.rvar.D, τr=τr_newton, τa=τa_newton, maxit=maxit_newton, verbose=verbose)

        # Below, turnon! checks if some of the newly calculated nonlasing modes have
        # Im{ω} > 0.  If some do, it picks the one with the most positive ω and turns it
        # on.
        #
        # Even if multiple nonlasing modes had Im{ω} > 0, turning on one mode increases the
        # hole-burning term and makes the system lossier, and therefore could make all the
        # remaining positive Im{ω} nonpositive again.  Therefore, we must not turn on all
        # the nonlasing modes with Im{ω} > 0, but do so only one-by-one.
        m_turnon = turnon!(lsol, nlsol)

        # If some mode was turned on above, continue the loop to perform solve_leq!
        # again to recalculate the lasing modes including the newly turned-on mode.
        #
        # If no mode was turned on, break the loop.
        m_turnon ≠ 0 || break
    end

    check_conflict(lsol, nlsol)

    return n_anderson, t_anderson
end


# Pump up or down the system to a target pump parameter.
# This changes the pump parameter while solving only the nonlasing equation.  Then, once the
# pump parameter reaches the target value, this function figures out which modes to turn
# on.  See Esterhazy et al.'s paper.
function pump!(lsol::LasingSol, lvar::LasingVar,
               nlsol::NonlasingSol, nlvar::NonlasingVar,
               param::SALTParam,
               dvec::AbsVecReal,  # trajectory of pump strength parameter to follow
               setD₀!::Function;  # setD₀!(param, d) sets pump strength param.D₀ corresponding to pump strength parameter d
               τr_newton::Real=TR_NEWTON,  # relative tolerance for Newton method to solve nonlasing equation
               τa_newton::Real=TA_NEWTON,  # absolute tolerance for Newton method to solve nonlasing equation
               maxit_newton::Integer=MAXIT_NEWTON,  # maximum number of Newton iteration steps
               m_anderson::Integer=M_ANDERSON,  # number of basis vectors to use in Anderson acceleration
               τr_anderson::Real=TR_ANDERSON,  # relative tolerance; consider using Base.rtoldefault(Float)
               τa_anderson::Real=TA_ANDERSON,  # absolute tolerance
               maxit_anderson::Integer=MAXIT_ANDERSON,  # maximum number of Anderson iteration steps
               verbose::Bool=true)
    println("\nPump nonlasinge equation.")
    for d = dvec[1:end-1]  # d = dvec[end] will be handled in solve_salt! below
        setD₀!(param, d)
        solve_nleq!(nlsol, nlvar, param, τr=τr_newton, τa=τa_newton, maxit=maxit_newton, verbose=verbose)
        verbose && println("d = $d, ωₙₗ = $(string(nlsol.ω)[17:end])")  # 17 is to skip header "Complex{Float64}"
    end

    println("\nFind lasing modes at pumped point.")
    solve_salt!(lsol, lvar, nlsol, nlvar, param, dvec[end], setD₀!,
                τr_newton=τr_newton, τa_newton=τa_newton, maxit_newton=maxit_newton,
                m_anderson=m_anderson, τr_anderson=τr_anderson, τa_anderson=τa_anderson, maxit_anderson=maxit_anderson,
                verbose=verbose)

    return nothing
end


# Simulate the laser operation as the pump parameter changes, by solving the lasing and
# nonlasing equations.
# Unlike pump!, this function solves both the lasing and nonlasing equations for every pump
# parameter.
function simulate!(lsol::LasingSol, lvar::LasingVar,
                   nlsol::NonlasingSol, nlvar::NonlasingVar,
                   param::SALTParam,
                   dvec::AbsVecReal,  # trajectory of pump strength parameter to follow
                   setD₀!::Function;  # setD₀!(param, d) sets pump strength param.D₀ corresponding to pump strength parameter d
                   outωaψ::NTuple{3,Bool}=(true,true,false),  # true to output ω, a, ψ
                   doutvec::AbsVecReal=dvec,  # output results when dvec[i] ∈ doutvec
                   τr_newton::Real=TR_NEWTON,  # relative tolerance for Newton method to solve nonlasing equation
                   τa_newton::Real=TA_NEWTON,  # absolute tolerance for Newton method to solve nonlasing equation
                   maxit_newton::Integer=MAXIT_NEWTON,  # maximum number of Newton iteration steps
                   m_anderson::Integer=M_ANDERSON,  # number of basis vectors to use in Anderson acceleration
                   τr_anderson::Real=TR_ANDERSON,  # relative tolerance; consider using Base.rtoldefault(Float)
                   τa_anderson::Real=TA_ANDERSON,  # absolute tolerance
                   maxit_anderson::Integer=typemax(Int),  # maximum number of Anderson iteration steps
                   verbose::Bool=true)
    # Create output storages.
    outω, outa, outψ = outωaψ
    nout = length(doutvec)

    M = length(lsol)
    N = length(lsol.ψ[1])

    ωout = outω ? MatComplex(undef,M,nout) : MatComplex(undef,M,0)
    aout = outa ? zeros(M,nout) : zeros(M,0)  # use zeros because some entries of a²out won't be overwritten
    ψout = outψ ? Array{CFloat,3}(undef,M,nout,N) : Array{CFloat,3}(undef,M,0,N)

    nAA = zeros(Int, nout)  # nummber of Anderson acceleration steps
    tAA = zeros(nout)  # time taken for Anderson acceleration

    println("\nStart SALT simulation.")
    cout = 1  # index of doutvec
    for d = dvec
        n_anderson, t_anderson =
            solve_salt!(lsol, lvar, nlsol, nlvar, param, d, setD₀!,
                        τr_newton=τr_newton, τa_newton=τa_newton, maxit_newton=maxit_newton,
                        m_anderson=m_anderson, τr_anderson=τr_anderson, τa_anderson=τa_anderson, maxit_anderson=maxit_anderson,
                        verbose=verbose)

        # Output results for current d.
        if d == doutvec[cout]
            nAA[cout] = n_anderson
            tAA[cout] = t_anderson

            # Output the lasing modes.
            m_active = lsol.m_active
            outω && (ωout[m_active,cout] .= lsol.ω[m_active])
            outa && (aout[m_active,cout] .= .√lsol.a²[m_active])
            if outψ
                for m = m_active
                    ψout[m,cout,:] .= lsol.ψ[m]
                end
            end

            # Output nonlasing modes.
            m_active = nlsol.m_active  # this does not conflict with lsol.m_active, because check_conflict() passed
            outω && (ωout[m_active,cout] .= nlsol.ω[m_active])
            if outψ
                for m = m_active
                    ψout[m,cout,:] .= nlsol.ψ[m]
                end
            end

            cout += 1
        end
    end

    return ωout, aout, ψout, nAA, tAA
end


# Find the threshold of the mode with a given mode index using the secant method.
#
# At the threshold d = dτ we seek to obtain, except for the mode (or modes, if degenerate)
# of interest, all the other modes are either sufficiently above their thresholds (and
# therefore lasing), or sufficiently below their thresholds (and therefore nonlasing).
#
# This method assumes that all the modes that should be lasing at d = dτ (except for the
# modes of interest that should be on set of lasing at d = dτ) are turned on at both d = dl
# and dr, where (dl, dr) = drng.  This is because the methed assumes that the only modes
# that turn on or shut down during the application of the quasi-Newton method are the modes
# of interest.
#
# This method solves only the lasing equation during the quasi-Newton iteration, but solves
# the nonlasing equation after finding the threshold to verify it in the nonlasing equation.
function find_threshold_secant!(lsol::LasingSol, lvar::LasingVar,
                                nlsol::NonlasingSol, nlvar::NonlasingVar,
                                param::SALTParam,
                                m::Integer,  # index of mode whose threshold is to be calculated
                                drng::Tuple{Real,Real},  # threshold is between drng[1] and drng[2]
                                setD₀!::Function;  # setD₀!(param, d) sets pump strength param.D₀ corresponding to pump strength parameter d
                                τr_threshold::Real=TR_THRESHOLD,  # relative tolerance of secant method
                                τa_threshold::Real=TA_THRESHOLD,  # absolute tolerance of secant method
                                maxit_threshold::Integer=MAXIT_THRESHOLD,  # maximum number of secant steps
                                τr_newton::Real=TR_NEWTON,  # relative tolerance for Newton method to solve nonlasing equation
                                τa_newton::Real=TA_NEWTON,  # absolute tolerance for Newton method to solve nonlasing equation
                                maxit_newton::Integer=MAXIT_NEWTON,  # maximum number of Newton iteration steps
                                m_anderson::Integer=M_ANDERSON,  # number of basis vectors to use in Anderson acceleration
                                τr_anderson::Real=TR_ANDERSON,  # relative tolerance; consider using Base.rtoldefault(Float)
                                τa_anderson::Real=TA_ANDERSON,  # absolute tolerance
                                maxit_anderson::Integer=MAXIT_ANDERSON,  # maximum number of Anderson iteration steps
                                verbose::Bool=true)
    check_conflict(lsol, nlsol)
    M = length(lsol)
    N = length(lsol.ψ[1])

    # Below, dl ≤ dr does not have to hold.  dr is just d examined later than dl in the
    # pump parameter scan outside this function.
    xₙ₋₁, xₙ = drng

    # Assume the threshold is between dl and dr.  This assumption is not necessary for the
    # secant method,
    println("\nStart secant method to find threshold of mode $m between d = $xₙ₋₁ and $xₙ.")

    # Solve the lasing equation at d = xₙ first, because lsol is likely to contain the guess
    # for d = xₙ.
    d = xₙ
    if !lsol.active[m]
        # Here, we don't use turnon! because we want to push the nonlasing mode m as a guess
        # for the lasing mode m even if it has Im{ωₙₗ} < 0.  (turnon! does not turn on such
        # a mode.)  In that case, upon solving the lasing equation, the lasing mode m will
        # have a² < 0, which is still used as a function value in the quasi-Newton method.
        push!(lsol, m, nlsol)
        pop!(nlsol, m)
    end
    @assert lsol.active[m]

    println("d = $d:")
    setD₀!(param, d)
    solve_leq!(lsol, lvar, param, m=m_anderson, τr=τr_anderson, τa=τa_anderson, maxit=maxit_anderson, verbose=verbose)
    fₙ = lsol.a²[m]


    # Solve the lasing equation at d = xₙ₋₁.  Note that lsol.active[m] == true now, so we
    # don't have to perform push!(lsol, m, nlsol).
    d = xₙ₋₁
    println("d = $d:")
    setD₀!(param, d)
    solve_leq!(lsol, lvar, param, m=m_anderson, τr=τr_anderson, τa=τa_anderson, maxit=maxit_anderson, verbose=verbose)
    fₙ₋₁ = lsol.a²[m]

    # Make xₙ closer to the root than xₙ₋₁.  This is unnecessary in Newton's method
    # where the slope is calculate at xₙ₊₁, but necessary in the quasi-Newton method where
    # the slope is approximated by (fₙ₊₁-fₙ)/(xₙ₊₁-xₙ).  xₙ₊₁ is obtained regardless of the
    # order of xₙ and xₙ₊₁ because it is the location where the line connecting (xₙ,fₙ) and
    # (xₙ₊₁,fₙ₊₁) crosses the horizontal axis.  However, (fₙ₊₁-fₙ)/(xₙ₊₁-xₙ) is different
    # from (fₙ₊₁-fₙ₋₁)/(xₙ₊₁-xₙ₋₁), and the latter would be a better approximation of
    # f′(xₙ₊₁) if |xₙ₊₁-xₙ₋₁| < |xₙ₊₁-xₙ|.
    if abs(fₙ) > abs(fₙ₋₁)
        (xₙ, xₙ₋₁) = (xₙ₋₁, xₙ)
        (fₙ, fₙ₋₁) = (fₙ₋₁, fₙ)
    end

    println("\nStart secant iteration.")
    lₙ₋₁ = l₀ = abs(fₙ₋₁)
    lₙ = abs(fₙ)
    τ = max(τr_threshold * l₀, τa_threshold)
    n_secant = 0
    while lₙ > τ
        # Calculate the quasi-Newton step.
        dfdx = (fₙ-fₙ₋₁) / (xₙ-xₙ₋₁)
        ∆x = -fₙ / dfdx

        # Update the previous approximate solution.
        xₙ₋₁ = xₙ
        fₙ₋₁ = fₙ
        lₙ₋₁ = lₙ

        # Update the current approximate solution.
        xₙ += ∆x
        d = xₙ
        println("d = $d:")
        setD₀!(param, d)
        solve_leq!(lsol, lvar, param, m=m_anderson, τr=τr_anderson, τa=τa_anderson, maxit=maxit_anderson, verbose=verbose)
        fₙ = lsol.a²[m]
        lₙ = abs(fₙ)

        println("After secant step $(n_secant+=1), mode $m has |aₗ²[$m]| = $lₙ.")

        if lₙ > lₙ₋₁
            println("|aₗ²[$m]| (= |f| of the secant method) increased.  Terminating the secant method.")

            # We do not execute the following, because later when solving the nonlasing
            # equation, param must contain the precalculated population inversion D for
            # the lasing solution, which is for the latest d at this point.  Simply
            # reverting d to the previous value will not revert D for this previous d.
            # println("|aₗ²[$m]| (= |f| of the secant method) increased.  Revert to previous solution.")
            # d = xₙ₋₁

            break
        end
    end
    println("Secant method terminated: d = $d")

    # Verify if Im{ωₙₗ[m]} ≈ 0.
    #
    # We need to solve the nonlasing equation only for the mode m.  So, empty all the modes
    # from the nonlasing solution first.
    println("\nVerify if Im{ωₙₗ[$m]} ≈ 0.  First, empty nonlasing solution:")
    for n = nlsol.m_active
        push!(lsol, n, nlsol)
        pop!(nlsol, n)
    end

    # Push the mode m to the nonlasing solution.
    # Here, we don't use shutdown! because we want to push the lasing mode m as a guess
    # for the nonlasing mode m even if it aₗ² > 0.  (shutdown! does not shut down such a
    # mode.)  This is to solve the nonlasing equation and see if ωₙₗ[m] has a close-to-
    # zero imaginary part.
    println("\nThen, Add mode $m to nonlasing solution:")
    push!(nlsol, m, lsol)
    pop!(lsol, m)

    # Solve the nonlasing equation for the mode m.
    println()
    solve_nleq!(nlsol, nlvar, param, lvar.rvar.D, τr=τr_newton, τa=τa_newton, maxit=maxit_newton, verbose=verbose)
    println("\nResult: Im{ωₙₗ[$m]} = $(imag(nlsol.ω[m]))")

    return d
end

# Find the threshold of the mode with a given mode index using the bisection method.
# Assume drng = (dl, dr) are both quite close to the threshold, and therefore the
# initial guess solution lsol and nlsol are good guesses for both d = dl and dr.
function find_threshold_bisect!(lsol::LasingSol, lvar::LasingVar,
                                nlsol::NonlasingSol, nlvar::NonlasingVar,
                                param::SALTParam,
                                m::Integer,  # index of mode whose threshold is to be calculated
                                drng::Tuple{Real,Real},  # threshold is between drng[1] and drng[2]
                                setD₀!::Function;  # setD₀!(param, d) sets pump strength param.D₀ corresponding to pump strength parameter d
                                τr_threshold::Real=TR_THRESHOLD,  # relative tolerance of bisection method
                                τa_threshold::Real=TA_THRESHOLD,  # absolute tolerance of bisection method
                                maxit_threshold::Integer=MAXIT_THRESHOLD,  # maximum number of bisection steps
                                τr_newton::Real=TR_NEWTON,  # relative tolerance for Newton method to solve nonlasing equation
                                τa_newton::Real=TA_NEWTON,  # absolute tolerance for Newton method to solve nonlasing equation
                                maxit_newton::Integer=MAXIT_NEWTON,  # maximum number of Newton iteration steps
                                m_anderson::Integer=M_ANDERSON,  # number of basis vectors to use in Anderson acceleration
                                τr_anderson::Real=TR_ANDERSON,  # relative tolerance; consider using Base.rtoldefault(Float)
                                τa_anderson::Real=TA_ANDERSON,  # absolute tolerance
                                maxit_anderson::Integer=MAXIT_ANDERSON,  # maximum number of Anderson iteration steps
                                verbose::Bool=true)
    check_conflict(lsol, nlsol)
    M = length(lsol)
    N = length(lsol.ψ[1])

    # Below, dl ≤ dr does not have to hold.  dr is just d examined later than dl in the
    # pump parameter scan outside this function.
    dl, dr = drng
    println("\nStart bisection method to find threshold of mode $m between d = $dl and $dr.")

    # Solve at d = dr first, because lsol is likely to contain the guess for d = dr, which
    d = dr
    solve_salt!(lsol, lvar, nlsol, nlvar, param, d, setD₀!,
                τr_newton=τr_newton, τa_newton=τa_newton, maxit_newton=maxit_newton,
                m_anderson=m_anderson, τr_anderson=τr_anderson, τa_anderson=τa_anderson, maxit_anderson=maxit_anderson,
                verbose=verbose)
    lase_r = lsol.active[m]  # true if lasing at d = dr
    lnl_r = islnl(lsol, nlsol, m)

    d = dl
    solve_salt!(lsol, lvar, nlsol, nlvar, param, d, setD₀!,
                τr_newton=τr_newton, τa_newton=τa_newton, maxit_newton=maxit_newton,
                m_anderson=m_anderson, τr_anderson=τr_anderson, τa_anderson=τa_anderson, maxit_anderson=maxit_anderson,
                verbose=verbose)
    lase_l = lsol.active[m]  # true if lasing at d = dl
    lnl_l = islnl(lsol, nlsol, m)

    n_bisect = 0
    println("After bisection step $n_bisect, lasing status of mode $m: $lase_l at d = $dl; $lase_r at d = $dr.")

    # In order to apply the search algorithm, the mode m must lase at only one of dl and dr.
    xor(lase_l, lase_r) || throw(ArgumentError("Must lase at one and only one of d = ($dl, $dr)."))

    while (n_bisect+=1) ≤ maxit_bisect && abs(dr-dl) > max(τr_bisect * abs(dr), τa_bisect)
        if lnl_r && lnl_l
            @warn "Bisection terminates prematurely because mode $m is seen simultaneously lasing and nonlasing at both dl = $dl and dr = $dr.  " *
                "Mode $m is too close to threshold for given solver tolerance.  Further bisection is meaningleses.  " *
                "To get more accurate threshold, lower solver tolerance."
            break
        end
        dc = 0.5(dl+dr)
        d = dc
        solve_salt!(lsol, lvar, nlsol, nlvar, param, d, setD₀!,
                    τr_newton=τr_newton, τa_newton=τa_newton, maxit_newton=maxit_newton,
                    m_anderson=m_anderson, τr_anderson=τr_anderson, τa_anderson=τa_anderson, maxit_anderson=maxit_anderson,
                    verbose=true)
        lase_c = lsol.active[m]  # true if lasing at d = dc
        lnl_c = islnl(lsol, nlsol, m)

        if lase_c == lase_l
            println("Setting dl = $dc.")
            dl = dc
            lase_l = lase_c
            lnl_l = lnl_c
        else
            println("Setting dr = $dc.")
            dr = dc
            lase_r = lase_c
            lnl_r = lnl_c
        end
        @assert xor(lase_l, lase_r)

        println("After bisection step $n_bisect, lasing status of mode $m: $lase_l at d = $dl; $lase_r at d = $dr.")
    end

    return 0.5(dl+dr)
end

find_threshold! = find_threshold_secant!
