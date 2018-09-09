using GeometryPrimitives
using StaticArrays

@testset "gainobj" begin

box_n = Box(-[0.5,0.5,0.5],[1,1,1])
gbox_n = GainObject(box_n, d::Real->-d)
@test (d = rand(); gbox_n.D₀fun(d) == -d)

box_p = Box([0.5,0.5,0.5],[1,1,1])
gbox_p = GainObject(box_p, d::Real->d)
@test (d = rand(); gbox_p.D₀fun(d) == d)

gobj_vec = [gbox_n, gbox_p]

# grid: from [-2,-2,-2] to [2,2,2] with ∆ = 1.  Exclude ghost points.
l = (([-2,-1,0,1],[-2,-1,0,1],[-2,-1,0,1]), ([-1.5,-0.5,0.5,1.5],[-1.5,-0.5,0.5,1.5],[-1.5,-0.5,0.5,1.5]))
N = SVector(4,4,4)

D₀ = zeros(3*prod(N))
d = 1.0
assign_pumpstr!(D₀, gobj_vec, d, N, l)

# Constrcut the expected D₀.
D₀exp = zeros(3, N...)

# x-components
D₀exp[1, 2, 2:3, 2:3] .= -1.0  # gbox_n
D₀exp[1, 3, 3:4, 3:4] .= 1.0  # gbox_p

# y-components
D₀exp[2, 2:3, 2, 2:3] .= -1.0  # gbox_n
D₀exp[2, 3:4, 3, 3:4] .= 1.0  # gbox_p

# z-components
D₀exp[3, 2:3, 2:3, 2] .= -1.0  # gbox_n
D₀exp[3, 3:4, 3:4, 3] .= 1.0  # gbox_p

@test D₀exp[:] == D₀

end  # @testset "gainobj"
