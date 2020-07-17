using Polynomials

@doc raw"""
Returns the result of the integral ``∫_τ^∞ t^k exp(-t^2 / σ) dt``
"""
function integral_monomial_gaussian(k::Integer, τ::T; σ=1, upper_bound_ok=false) where T
    @assert k ≥ -4
    @assert σ > 0
    τdσ = τ / T(sqrt(σ))  # τ divided by σ

    ## Even k
    if k == -4
        exp(-τdσ^2) / 3τ^3 - 2 / T(3σ) * integral_monomial_gaussian(-2, τ, σ=σ)
    elseif k == -2
        sqrt(T(π) / σ)        * (erf(τdσ) - 1) + exp(-τdσ^2) / τ
    elseif k == 0
        sqrt(T(π) * σ)    / 2 * (1 - erf(τdσ))
    elseif k == 2
        sqrt(T(π) * σ^3)  / 4 * (1 - erf(τdσ)) + (                             σ * τ / 2) * exp(-τdσ^2)
    elseif k == 4
        3sqrt(T(π) * σ^5) / 8 * (1 - erf(τdσ)) + (             σ * τ^3 / 2 +  3σ^2*τ / 4) * exp(-τdσ^2)
    elseif k == 6
        15sqrt(T(π) * σ^7)/16 * (1 - erf(τdσ)) + (σ*τ^5 / 2 + 5σ^2*τ^3 / 4 + 15σ^3*τ / 8) * exp(-τdσ^2)
    #
    ## Odd k
    elseif k == -3
        upper_bound_ok || error("The case k == -3 is not implemented.")
        # ∫_τ^∞ (1/t^3) * exp(-t^2 / σ) dt ≤ (1/τ^3) ∫_τ^∞ exp(-t^2 / σ) dt
        integral_monomial_gaussian(0, τ, σ=σ) / τ^3
    elseif k == -1
        upper_bound_ok || error("The case k == -1 is not implemented.")
        # ∫_τ^∞ (1/t) * exp(-t^2 / σ) dt ≤ (1/τ) ∫_τ^∞ exp(-t^2 / σ) dt
        integral_monomial_gaussian(0, τ, σ=σ) / τ
        # Better bound:
        # \intx^\infty t^{-1} e^{-t^2} dt is
        # = \int{x^2} s^{-1} e^{-s} ds (use the change of variable s=t^2) is e^{-x^2}/(2x^2)
    elseif k == 1
        (                                 σ) / 2 * exp(-τdσ^2)
    elseif k == 3
        (                   σ * τ^2 +   σ^2) / 2 * exp(-τdσ^2)
    elseif k == 5
        (         σ * τ^4 + 2σ^2*τ^2 + 2σ^3) / 2 * exp(-τdσ^2)
    elseif k == 7
        (σ*τ^6 + 3σ^2*τ^4 + 6σ^3*τ^2 + 6σ^4) / 2 * exp(-τdσ^2)
    #
    ## Fallback by recursion
    else
        σ * τ^(k - 1) / 2 * exp(-τdσ^2) + σ * (k - 1) / 2 * integral_monomial_gaussian(k - 2, τ, σ=σ)
    end
end

@doc raw"""
Returns the result of the integral ``∫_τ^∞ t^k P(t) exp(-t^2 / σ) dt``
where `P` is a polynomial in `t`
"""
function integral_polynomial_gaussian(P::Polynomial, τ::T; k=0, σ=1, upper_bound_ok=false) where T
    # -1 translates from index in coeff to power in t
    sum(coeff * integral_monomial_gaussian(i + k - 1, τ, σ=σ, upper_bound_ok=upper_bound_ok)
        for (i, coeff) in enumerate(P.a) if coeff != 0)
end

@doc raw"""
Computes an upper bound for the sum ``∑_{|G| > G0} |β G|^k Q(|β G|) exp(-|β G|^2 / σ)``
where all `G` are on a lattice defined by `recip_lattice`, `β` and `σ` are a positive
constants, `k` is integer and `Q` is a polynomial both chosen such that
``t^k Q(t) exp(-t / σ)`` is a decreasing function for `t > G0`.
"""
function bound_sum_polynomial_gaussian(polynomial::Polynomial{T}, recip_lattice, G0::T;
                                       β=1, σ=1, k=0, Gmin=G0) where {T}
    # Determine dimensionality: Note: Clashes with usual DFTK convention
    @assert !iszero(recip_lattice[:, end])
    m = size(recip_lattice, 1)
    @assert size(recip_lattice) == (m, m)

    # Diameter of recip_lattice unit cell.
    diameter = norm(recip_lattice * ones(m))
    @assert diameter ≥ 0  # We assume lattice to be positively orianted
    @assert Gmin > 0
    @assert G0 ≤ Gmin

    # The terms (depending on β, n and G) we sum over.
    term(β, n, G) = (β * G)^n * polynomial(β * G) * exp(-β*G^2)

    if m == 1
        a = abs(recip_lattice[1, 1])
        start = floor(G0 / a) * a
        @assert start ≤ Gmin
        intm = integral_polynomial_gaussian(polynomial, β * Gmin, k=k, σ=σ,
                                            upper_bound_ok=true) / T(β)
        return  2 / a * ((Gmin - start) * term(β, k, Gmin) + intm)
    elseif m == 2 || m == 3
        prefactor = (m == 2 ? 2T(π) : 4T(π)) / abs(det(recip_lattice))
        start = G0 - diameter

        intm = integral_polynomial_gaussian(polynomial, β * Gmin, σ=σ,
                                            k=k + m - 1, upper_bound_ok=true) / T(β)^m
        result = prefactor * ((Gmin - start) * term(β, k + m - 1, Gmin) + intm)

        for (j, aj) in enumerate(eachcol(recip_lattice))
            newlattice = []
            nRj = 1
            for (k, ak) in enumerate(eachcol(recip_lattice))
                k == j && continue
                angle = dot(aj, ak) / dot(aj, aj)

                nRj = nRj * (2 + ceil(abs(angle)))
                newlatticevector = Vector(ak - angle * aj)
                deleteat!(newlatticevector, j)
                push!(newlattice, newlatticevector)
            end
            newlattice = hcat(newlattice...)

            result += nRj * bound_sum_polynomial_gaussian(polynomial, newlattice,
                                                          G0 - diameter, σ=σ, β=β, k=k, Gmin=Gmin)
        end

        return result
    end
    error("Dimensionality > 3")
end
