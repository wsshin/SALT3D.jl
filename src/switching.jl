export turnon!, shutdown!, check_conflict

# Nonlasing -> Lasing: take the mth nonlasing mode and use it to set the guess information
# about the mth lasing mode.
function Base.push!(lsol::LasingSol, m::Integer, nlsol::NonlasingSol)
    lsol.ω[m] = real(nlsol.ω[m])
    lsol.a²[m] = 0
    iₐ = lsol.iₐ[m] = nlsol.iₐ[m]
    ψ = lsol.ψ[m] .= nlsol.ψ[m]
    assert(ψ[iₐ] == 1)  # make sure ψₘ is already normalized

    M = length(lsol)
    lsol.act[m] = true
    lsol.m_act = (1:M)[lsol.act]

    return nothing
end

function Base.pop!(lsol::LasingSol, m::Integer)
    lsol.a²[m] = 0  # indicate this mode is nonlasing
    lsol.ψ[m] .= 0  # good for compressing data when writing in file

    M = length(lsol)
    lsol.act[m] = false
    lsol.m_act = (1:M)[lsol.act]

    return nothing
end


# Lasing -> Nonlasing: take the mth lasing mode and use it to set the guess information
# about the mth nonlasing mode.
function Base.push!(nlsol::NonlasingSol, m::Integer, lsol::LasingSol)
    nlsol.ω[m] = lsol.ω[m]  # scalar
    iₐ = nlsol.iₐ[m] = lsol.iₐ[m]  # scalar
    ψ = nlsol.ψ[m] .= lsol.ψ[m]  # vector
    assert(ψ[iₐ] == 1)  # make sure ψₘ is already normalized

    M = length(nlsol)
    nlsol.act[m] = true
    nlsol.m_act = (1:M)[nlsol.act]

    return nothing
end

function Base.pop!(nlsol::NonlasingSol, m::Integer)
    nlsol.ω[m] = real(nlsol.ω[m])  # indicate this mode is lasing
    nlsol.ψ[m] .= 0  # good for compressing data when writing in file

    M = length(nlsol)
    nlsol.act[m] = false
    nlsol.m_act = (1:M)[nlsol.act]

    return nothing
end


# Prepare LasingSol by turning on the nonlasing mode with the largest positive imaginary
# part, if any.  See Sec. III.D of the SALT paper.
# Return true if there is a mode to turn on.
function turnon!(lsol::LasingSol, nlsol::NonlasingSol)
    m = indmax(imag, nlsol.ω, nlsol.m_act)
    hasmode2turnon = m≠0
    if hasmode2turnon
        hasmode2turnon = imag(nlsol.ω[m]) > 0  # consider imag(ω) = 0 as nonlasing in order to keep lasing equations minimal
        if hasmode2turnon
            println("turn on: mode $m in ω ₙₗ = $(nlsol.ω)")
            push!(lsol, m, nlsol)
            pop!(nlsol, m)
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
        hasmode2shutdown = lsol.a²[m] ≤ 0  # consider a² = 0 as nonlasing in order to keep lasing equations minimal
        if hasmode2shutdown
            println("shut down: mode $m in a²ₗ = $(lsol.a²)")
            push!(nlsol, m, lsol)
            pop!(lsol, m)
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
