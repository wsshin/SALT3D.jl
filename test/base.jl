@testset "base" begin

@testset "argmax, argmin" begin
    v = Float64[11, 13, 14, 10, 12]

    # Test the behavior for a nonempty index vector.
    indv = [1,2,5]
    @test @inferred(argmax(identity, v)) == 3
    @test @inferred(argmax(identity, v, indv)) == 2
    @test @inferred(argmin(identity, v)) == 4
    @test @inferred(argmin(identity, v, indv)) == 1

    # Test the behavior for an empty index vector.
    indv = Int[]
    @test @inferred(argmax(identity, v, indv)) == 0
    @test @inferred(argmin(identity, v, indv)) == 0

end  # @testset "argmax, argmin"

end  # @testset "base"
