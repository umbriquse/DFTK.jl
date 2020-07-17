using Test
using DFTK: integral_monomial_gaussian, bound_sum_polynomial_gaussian
using QuadGK
include("testcases.jl")

function integral_monomial_gaussian_by_recursion(k::Integer, τ::T; σ=1, upper_bound_ok=false) where T
    if k == 1 || k == -4
        integral_monomial_gaussian(k, τ, σ=σ, upper_bound_ok=false)
    else
        (σ * τ^(k - 1) / 2 * exp(-τdσ^2) + σ * (k - 1) / 2
           * integral_monomial_gaussian(k - 2, τ, σ=σ, upper_bound_ok=upper_bound_ok))
    end
end

@testset "Test integral_monomial_gaussian" begin
    τ = abs(1 + 2rand())
    σ = 0.5 + 0.5*rand()

    @test integral_monomial_gaussian(1, τ, σ=σ)  ≈ quadgk(t -> t * exp(-t^2 / σ), τ, 100)
    @test integral_monomial_gaussian(-4, τ, σ=σ) ≈ quadgk(t -> exp(-t^2 / σ) / t^4, τ, 100)
    for k in append!([-4, -2], 0:20)
        @test(  integral_monomial_gaussian_by_recursion(k, τ, σ=σ)
              ≈ integral_monomial_gaussian(k, τ, σ=σ))
    end

    @test (  quadgk(t -> exp(-t^2 / σ) / t, τ, 100)
           ≤ integral_monomial_gaussian(-1, τ, σ=σ, upper_bound_ok=true))
    @test (  quadgk(t -> exp(-t^2 / σ) / t^3, τ, 100)
           ≤ integral_monomial_gaussian(-3, τ, σ=σ, upper_bound_ok=true))
end


@testset "Test bound_sum_polynomial_gaussian" begin
    psp = load_psp("hgh/lda/si-q4")
    recip_lattice = 2π * inv(silicon.lattice')

    T = Float64
    iproj = 2
    l = 0
    Q = DFTK.psp_projection_radial_polynomial(T, psp, iproj, l)^2

    function naive_sum(P, recip_lattice, G0; Nmax=50, β=1, σ=1, k=0)
        res = zero(eltype(recip_lattice))
        for i = -Nmax:Nmax, j = -Nmax:Nmax, k = -Nmax:Nmax
            G = recip_lattice * Vec3(i, j , k)
            if norm(G) ≥ G0
                res += (β * norm(G))^k * P(β * norm(G)) * exp(-norm(β * G)^2 / σ)
            end
        end
        res
    end

    bound_diff(x; β=1, σ=1, k=0) = (
          bound_sum_polynomial_gaussian(Q, recip_lattice, x, β=β, σ=σ, k=k)
        - naive_sum(Q, recip_lattice, x, β=β, σ=σ, k=k)
    )

    @test bound_diff(3.0) ≥ -1e-16
    @test bound_diff(4.5) ≥ -1e-16
    @test bound_diff(5.0) ≥ -1e-16
    @test bound_diff(10.0) ≥ -1e-16
end
