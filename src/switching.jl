export turnon!, shutdown!, check_conflict

# Prepare LasingSol by turning on the nonlasing mode with the largest positive imaginary
# part, if any.  See Sec. III.D of the SALT paper.
# Return true if there is a mode to turn on.
function turnon!(lsol::LasingSol, nlsol::NonlasingSol)
    m = indmax(imag, nlsol.ω, nlsol.m_act)
    hasmode2turnon = m≠0
    if hasmode2turnon
        ω = nlsol.ω[m]
        hasmode2turnon = imag(ω) > 0
        if hasmode2turnon  # consider imag(ω) = 0 as nonlasing in order to keep lasing equation minimal
            # Set guess values for the lasing mode
            lsol.ω[m] = real(ω)
            lsol.a²[m] = 0.0

            iₐ = nlsol.iₐ[m]
            lsol.iₐ[m] = iₐ

            ψ = nlsol.ψ[m]
            lsol.ψ[m] .= ψ ./ ψ[iₐ]  # make ψ[iₐ] = 1

            push!(lsol, m)
            pop!(nlsol, m)
            println("turned on: m = $m in ω ₙₗ = $(nlsol.ω) with nonlasing m = $(nlsol.m_act)")
        end
    end

    return hasmode2turnon
end


# Prepare NonlasingSol by shutting down the lasing mode with negative amplitude, if any.
# Return true if there is any mode to shut down.
function shutdown!(lsol::LasingSol, nlsol::NonlasingSol)
    m = indmin(identity, lsol.a², lsol.m_act)
    hasmode2shutdown = m≠0
    if hasmode2shutdown
        a² = lsol.a²[m]
        hasmode2shutdown = a² ≤ 0
        if hasmode2shutdown  # consider a² = 0 as nonlasing in order to keep lasing equation minimal
            nlsol.ω[m] = lsol.ω[m]

            iₐ = lsol.iₐ[m]
            nlsol.iₐ[m] = iₐ

            ψ = lsol.ψ[m]
            nlsol.ψ[m] .= ψ ./ ψ[iₐ]  # make ψ[iₐ] = 1

            push!(nlsol, m)
            pop!(lsol, m)
            println("shut down: m = $m in a²ₗ = $(lsol.a²) with lasing m = $(lsol.m_act)")
        end
    end

    return hasmode2shutdown
end


# Once switching has occurred, lasing and nonlasing modes are re-calculated, and then they
# must pass this test.
function check_conflict(lsol, nlsol)
    M = length(lsol)
    length(nlsol) == M ||
        throw(ArgumentError("length(lsol) = $M and length(nlsol) = $(length(nlsol)) must be the same."))

    for m = 1:M
        xor(lsol.act[m], nlsol.act[m]) ||
            throw(ArgumentError("Mode $m must be either lasing or nonlasing, exclusively."))
    end

    return nothing
end
