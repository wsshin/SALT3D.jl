function Base.indmax(f::Function, v::AbsVec)
    ind = 0  # return 0 if v is empty
    val = -Inf
    for n = 1:length(v)
        if f(v[n]) ≥ val  # ≥ rather than > to prevent returning 0 when v has -Inf
            val = f(v[n])
            ind = n
        end
    end

    return ind
end

function Base.indmax(f::Function, v::AbsVec, indv::AbsVecInteger)
    ind = 0  # return 0 if v is empty
    val = -Inf
    for n = indv
        if f(v[n]) ≥ val  # ≥ rather than > to prevent returning 0 when v has -Inf
            val = f(v[n])
            ind = n
        end
    end

    return ind
end

function Base.indmin(f::Function, v::AbsVec)
    ind = 0  # return 0 if v is empty
    val = Inf
    for n = 1:length(v)
        if f(v[n]) ≤ val  # ≤ rather than < to prevent returning 0 when v has -Inf
            val = f(v[n])
            ind = n
        end
    end

    return ind
end

function Base.indmin(f::Function, v::AbsVec, indv::AbsVecInteger)
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
