using GeometryPrimitives

@testset "gainobj" begin

box = Box([0,0,0], [1,1,1])
gainbox = GainObject(box)
@test (d = rand(); gainbox.D₀fun(d) == d)





end  # @testset "gainobj"
