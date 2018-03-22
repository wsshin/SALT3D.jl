export GainObject
export assign_gainobj!

# GainObject has
# - Shape
# - scale factor to be multiplied to the true/false array.
#
# Later, to put an object with a spatially changing pump strength, I will be able to define
# a new GainObject.  Different types of GainObjects will need to define different assignment
# function.  Eventually, I will need to write something like MaxwellFDM's assign function
# such that it internally uses the individual shape's assign function.  But let's not think
# about that right now.
#
# This may require defining GainObject as an abstract type later, and defining its concrete
# types.

mutable struct GainObject{K,S<:Shape,F<:Function}  # use S<:Shape{K} after Julia issue #26321 is fixed
    shape::S
    D₀fun::F  # population inversion in absence of lasing; function of pump parameter d
    GainObject{K,S,F}(shape, D₀fun) where {K,S<:Shape{K},F} = new(shape, D₀fun)
end

GainObject(shape::S, D₀fun::F) where {K,S<:Shape{K},F<:Function} = GainObject{K,S,F}(shape, D₀fun)
GainObject(shape::Shape) = GainObject(shape, d::Real->d)

function assign_gainobj!(D₀::AbsVecReal, gobj_vec::AbsVec{<:GainObject{3}}, d::Real, N::SVector{3,Int}, l::MaxwellFDM.Tuple23{AbsVecReal})
    D₀array = reshape(D₀, 3, N.data...)
    gt = PRIM
    gt_cmp₀ = SVector(gt, gt, gt)
    for nw = nXYZ
        gt_cmp = broadcast((k,w,g)->(k==w ? alter(g) : g), nXYZ, nw, gt_cmp₀)
        lcmp = t_ind(l,gt_cmp)
        D₀cmp = @view D₀array[nw,:,:,:]
        for gobj = gobj_vec
            shape = gobj.shape
            D₀fun = gobj.D₀fun
            D₀val = D₀fun(d)
            assign_val_shape!(D₀cmp, D₀val, shape, lcmp)
        end
    end

    return nothing
end
