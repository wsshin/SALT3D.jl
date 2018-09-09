# Return the index of the maximum entry of a given vector transformed by f.
Base.argmax(f::Function, v::AbsVec) = argmax(f, v, 1:length(v))

# Among specified indices, return the index of the maximum entry of a given vector
# transformed by f.
function Base.argmax(f::Function, v::AbsVec, indv::AbsVecInteger)
    ind = 0  # return 0 if indv is empty
    val = -Inf
    for n = indv
        if f(v[n]) ≥ val  # ≥ rather than > to prevent returning 0 when v has -Inf
            val = f(v[n])
            ind = n
        end
    end

    return ind
end


# Return the index of the minimum entry of a given vector transformed by f
Base.argmin(f::Function, v::AbsVec) = argmin(f, v, 1:length(v))

# Among specified indices, return the index of the minimum entry of a given vector
# transformed by f.
function Base.argmin(f::Function, v::AbsVec, indv::AbsVecInteger)
    ind = 0  # return 0 if v is empty
    val = Inf
    for n = indv
        if f(v[n]) ≤ val  # ≤ rather than < to prevent returning 0 when v has -Inf
            val = f(v[n])
            ind = n
        end
    end

    return ind
end
