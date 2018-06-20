@testset "base" begin

@testset "indmax, indmin" begin
    v = Float64[11, 13, 14, 10, 12]

    # Test the behavior for a nonempty index vector.
    indv = [1,2,5]
    @test @inferred(indmax(identity, v)) == 3
    @test @inferred(indmax(identity, v, indv)) == 2
    @test @inferred(indmin(identity, v)) == 4
    @test @inferred(indmin(identity, v, indv)) == 1

    # Test the behavior for an empty index vector.
    indv = Int[]
    @test @inferred(indmax(identity, v, indv)) == 0
    @test @inferred(indmin(identity, v, indv)) == 0

end  # @testset "indmax, indmin"

end  # @testset "base"
