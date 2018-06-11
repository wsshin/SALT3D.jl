export turnon!, shutdown!, check_conflict

# Nonlasing -> Lasing: take the mth nonlasing mode and use it to set the guess information
# about the mth lasing mode.
function Base.push!(lsol::LasingSol, m::Integer, nlsol::NonlasingSol)
    lsol.ω[m] = real(nlsol.ω[m])
    lsol.a²[m] = 0
    iₐ = lsol.iₐ[m] = nlsol.iₐ[m]
    ψ = lsol.ψ[m] .= nlsol.ψ[m]

    # Below, we don't test if ψ[iₐ] == 1, because the normalization of ψ was done by complex
    # division.  For a complex scalar z, z/z may not be exactly zero in floating-point
    # arithmetic, because z/w is basically evaluated as z*conj(w) / abs(w)^2, during which
    # rounding that could make z / z different from 1 occurs.
    assert(ψ[iₐ] ≈ 1)  # make sure ψₘ is already normalized

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

    # Below, we don't test if ψ[iₐ] == 1, because the normalization of ψ was done by complex
    # division.  For a complex scalar z, z/z may not be exactly zero in floating-point
    # arithmetic, because z/w is basically evaluated as z*conj(w) / abs(w)^2, during which
    # rounding that could make z / z different from 1 occurs.
    assert(ψ[iₐ] ≈ 1)  # make sure ψₘ is already normalized

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
# Return the index of the mode to turn on; return 0 otherwise.
function turnon!(lsol::LasingSol, nlsol::NonlasingSol)
    m = indmax(imag, nlsol.ω, nlsol.m_act)  # m==0 if nlsol.m_act is empty
    m≠0 && imag(nlsol.ω[m])>0 || return 0  # consider imag(ω) = 0 as nonlasing in order to keep lasing equations minimal

    # Now m ≠ 0 and imag(nlsol.ω[m]) > 0.
    print_with_color(:blue, "Turning on mode $m where ωₙₗ[m=1:$(length(nlsol))] = $(string(nlsol.ω)[17:end])...  ")  # 17 is to skip header "Complex{Float64}"
    push!(lsol, m, nlsol)
    pop!(nlsol, m)
    print_with_color(:blue, "Done!"); println()

    return m
end


# Prepare NonlasingSol by shutting down the lasing mode with negative amplitude, if any.
# Return the index of the mode to shut down; return 0 otherwise.
function shutdown!(lsol::LasingSol, nlsol::NonlasingSol)
    m = indmin(identity, lsol.a², lsol.m_act)  # m==0 if lsol.m_act is empty
    m≠0 && lsol.a²[m]≤0 || return 0  # consider a² = 0 as nonlasing in order to keep lasing equations minimal

    # Now m ≠ 0 and lsol.a²[m] ≤ 0.
    print_with_color(:blue, "Shutting down mode $m where aₗ²[m=1:$(length(lsol))] = $(lsol.a²)...  ")
    push!(nlsol, m, lsol)
    pop!(lsol, m)
    print_with_color(:blue, "Done!"); println()

    return m
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
