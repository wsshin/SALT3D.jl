export turnon!, shutdown!, check_conflict

# Nonlasing -> Lasing: take the mth nonlasing mode and use it to set the guess information
# about the mth lasing mode.
function Base.push!(lsol::LasingSol, m::Integer, nlsol::NonlasingSol, msgprefix::String="  ")
    print_with_color(:blue, msgprefix * "Pushing nonlasing mode $m with ωₙₗ[$m] = $(nlsol.ω[m]) to lasing solution.\n")
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
    lsol.activated[m] = true
    lsol.active[m] = true
    lsol.m_active = (1:M)[lsol.active]

    return nothing
end

function Base.pop!(lsol::LasingSol, m::Integer, msgprefix::String="  ")
    print_with_color(:blue, msgprefix * "Popping lasing mode $m from lasing solution.\n")
    lsol.a²[m] = 0  # indicate this mode is nonlasing
    lsol.ψ[m] .= 0  # good for compressing data when writing in file

    M = length(lsol)
    lsol.active[m] = false
    lsol.m_active = (1:M)[lsol.active]

    return nothing
end


# Lasing -> Nonlasing: take the mth lasing mode and use it to set the guess information
# about the mth nonlasing mode.
function Base.push!(nlsol::NonlasingSol, m::Integer, lsol::LasingSol, msgprefix::String="  ")
    print_with_color(:blue, msgprefix * "Pushing lasing mode $m with ωₗ[$m] = $(lsol.ω[m]) to nonlasing solution.\n")
    nlsol.ω[m] = lsol.ω[m]  # real scalar
    iₐ = nlsol.iₐ[m] = lsol.iₐ[m]  # scalar
    ψ = nlsol.ψ[m] .= lsol.ψ[m]  # vector

    # Below, we don't test if ψ[iₐ] == 1, because the normalization of ψ was done by complex
    # division.  For a complex scalar z, z/z may not be exactly zero in floating-point
    # arithmetic, because z/w is basically evaluated as z*conj(w) / abs(w)^2, during which
    # rounding that could make z / z different from 1 occurs.
    assert(ψ[iₐ] ≈ 1)  # make sure ψₘ is already normalized

    M = length(nlsol)
    nlsol.activated[m] = true
    nlsol.active[m] = true
    nlsol.m_active = (1:M)[nlsol.active]

    return nothing
end

function Base.pop!(nlsol::NonlasingSol, m::Integer, msgprefix::String="  ")
    print_with_color(:blue, msgprefix * "Popping nonlasing mode $m from nonlasing solution.\n")
    nlsol.ω[m] = real(nlsol.ω[m])  # indicate this mode is lasing
    nlsol.ψ[m] .= 0  # good for compressing data when writing in file

    M = length(nlsol)
    nlsol.active[m] = false
    nlsol.m_active = (1:M)[nlsol.active]

    return nothing
end


# Prepare LasingSol by turning on the nonlasing mode with the largest positive imaginary
# part, if any.  See Sec. III.D of the SALT paper.
# Return the index of the mode to turn on; return 0 otherwise.
function turnon!(lsol::LasingSol, nlsol::NonlasingSol, msgprefix::String="  ")
    m = indmax(imag, nlsol.ω, nlsol.m_active)  # m==0 if nlsol.m_active is empty
    m≠0 && imag(nlsol.ω[m])>0 || return 0  # consider imag(ω) = 0 as nonlasing in order to keep lasing equations minimal

    # Now m ≠ 0 and imag(nlsol.ω[m]) > 0.
    # If the mode m was shut down just now, do not turn it on; otherwise, turn on the mode.
    print_with_color(:blue, msgprefix * "Turning on mode $m where ωₙₗ[m=1:$(length(nlsol))] = $(string(nlsol.ω)[17:end])...\n")  # 17 is to skip header "Complex{Float64}"
    if nlsol.activated[m]  # mode m is just shut down
        nlsol.ω[m] = real(nlsol.ω[m])
        print_with_color(:red, msgprefix * "Don't, because mode $m was shut down just now.\n")
        return 0
    else
        push!(lsol, m, nlsol)
        pop!(nlsol, m)
        print_with_color(:blue, msgprefix * "Done.\n")
        return m
    end
end


# Prepare NonlasingSol by shutting down the lasing mode with negative amplitude, if any.
# Return the index of the mode to shut down; return 0 otherwise.
function shutdown!(lsol::LasingSol, nlsol::NonlasingSol, msgprefix::String="  ")
    m = indmin(identity, lsol.a², lsol.m_active)  # m==0 if lsol.m_active is empty
    m≠0 && lsol.a²[m]≤0 || return 0  # consider a² = 0 as nonlasing in order to keep lasing equations minimal

    # Now m ≠ 0 and lsol.a²[m] ≤ 0.
    # If the mode m was turned on just now, do not shut it down; otherwise, shut down the mode.
    print_with_color(:blue, msgprefix * "Shutting down mode $m where aₗ²[m=1:$(length(lsol))] = $(lsol.a²)...\n")
    if lsol.activated[m]  # mode m is just turned on
        lsol.a²[m] = 0
        print_with_color(:red, msgprefix * "Don't, because mode $m was turned on just now.\n")
        return 0
    else
        push!(nlsol, m, lsol)
        pop!(lsol, m)
        print_with_color(:blue, msgprefix * "Done.\n")
        return m
    end
end


# Once switching has occurred, lasing and nonlasing modes are re-calculated, and then they
# must pass this test.
function check_conflict(lsol, nlsol)
    M = length(lsol)
    length(nlsol) == M ||
        throw(ArgumentError("length(lsol) = $M and length(nlsol) = $(length(nlsol)) must be the same."))

    for m = 1:M
        xor(lsol.active[m], nlsol.active[m]) ||
            throw(ArgumentError("Mode $m must be either lasing or nonlasing, exclusively."))
    end

    return nothing
end

# Return true if the mode m is simultaneously lasing and nonlasing (i.e., at threshold).
#
# Even in such a case, the mode m is indicated only lasing or nonlasing, not both. (In other
# words, only one of lsol.active[m] and nlsol.active[m] is true.)  For example, if the mode
# m is turned on but its lasing amplitude is still zero, this mode is physically at
# threshold and therefore simultaneously lasing and nonlasing.  However, we have
# lsol.active[m] == true and nlsol.active[m] == false, because technically it is turned on.
# Similarly, if the mode is shut down but its eigenfrequency has a zero imaginary part, this
# mode is again physically at threshold and therefor simultaneously lasing and nonlasing,
# but we have lsol.active[m] == false and nlsol.active[m] == true because technically this
# mode is shut down.
islnl(lsol, nlsol, m) = lsol.a²[m]==0 && imag(nlsol.ω[m])==0
