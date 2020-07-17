
###
### Bounds for \|V P_{B_1}^⟂\|_op (potential_perpb1)
###

function bound_potential_perpb1_Vloc(hamblock, hamblock_Ecut2)
    basis2 = hamblock_Ecut2.basis
    kpoint2 = hamblock_Ecut2.kpoint
    model = basis2.model
    Ecut2 = basis2.Ecut
    kpoint = hamblock.kpoint
    T = eltype(basis2)
    @assert kpoint.coordinate == kpoint2.coordinate

    accu = zero(T)
    for (element, positions) in model.atoms
        if element isa ElementPsp
            @assert element.psp isa PspHgh

            # We need a bound for Σ_{G, |G| > qcut} |Q(t)| exp(-t^2 / 2) / t^2
            # where t = psp.rloc * |G|. Since |Q| is a decaying function not going
            # through zero after |G| > qcut, we may replace |Q| by Q or -Q depending
            # on the sign at qcut.
            Qloc = DFTK.psp_local_polynomial(T, element.psp)
            qcut = sqrt(2Ecut2)
            rts = [real.(rt) for rt in roots(Qloc) if imag(rt) < 1e-10]
            @assert length(rts) == 0 || maximum(rts) < qcut
            sign(Qloc(qcut)) < 0 && (Qloc = -Qloc)
            R = bound_sum_polynomial_gaussian(Qloc, model.recip_lattice, qcut,
                                              β=element.psp.rloc, k=-2, σ=2)
        else
            @assert element isa ElementCohenBergstresser
            R = 0  # assume Ecut2 to be large enough
        end
        accu += length(positions) * R / model.unit_cell_volume
    end

    # For the remaining term (inside B2 ∩ B1^⟂) we use an estimate precomputed
    # with the function bound_potential_b2_Vloc_b2_infty at a very large value
    # of Ecut for the employed basis in the hamblock. This yields a value for
    # |P_B V P_B |_\intfy norm for a basis B which is guaranteed to be larger than
    # B2 and thus an upper bound to the term |P_{B2} V P_{B2} |_\intfy.
    #
    # NOTE: These values are specific to the lattice and atomic positions
    #       we employ here and are not transferable.
    @assert basis2.Ecut ≤ 5000  # Values precomputed at Ecut=5000
    if is_cohen_bergstresser(model)
        bound_Vinf = 0.6681312576847188
    elseif is_linear_atomic(model)
        bound_Vinf = 7.230556115103383
    else
        error("Not implemented")
    end

    accu + bound_Vinf
end

@doc raw"""
Compute ``\|P_{B} Vloc P_{B}\|_\infty`` for ``B`` being the basis used in `hamblock`.
"""
function bound_potential_b2_Vloc_b2_infty(hamblock)
    basis = hamblock.basis
    model = basis.model
    T = eltype(basis)
    idx_local = only(findall(t -> t isa AtomicLocal, model.term_types))

    # We compute the supremum of the real-space values we know
    # (shifting the potential implicitly, which we know does not change the gap)
    Vloc_extrema = extrema(hamblock.operators[idx_local].potential)
    extent = (Vloc_extrema[2] - Vloc_extrema[1]) / T(2)

    # Then we use a gradient correction for the fact that we do not know the values
    # inside the cell. If g is the gradient vector and δ the cell diameter than the
    # possible addition is <g|d>/2 ≤ ||g|| δ / 2 where d is any lattice vector.
    # In turn we estimate ||g||_2 ≤ ||g||_∞ = ||\hat{g}||_1 i.e. by the l1-norm of
    # the Fourier coefficients of the gradient, which is |G| times the Fourier coefficients
    # of the potential itself.

    # local_potential_fourier is the actual FT of the real-space potential
    potterm(el, r, Gcart) = Complex{T}(DFTK.local_potential_fourier(el, norm(Gcart))
                                       * cis(-dot(Gcart, model.lattice * r)))

    # sqrt(Ω) because of normalised basis used in DFTK
    pot(G) = sum(potterm(elem, r, model.recip_lattice * G) / sqrt(model.unit_cell_volume)
                 for (elem, positions) in model.atoms
                 for r in positions)

    # Another sqrt(Ω) from going to the l1-norm of the Fourier coefficients
    sqrtΩ = sqrt(model.unit_cell_volume)
    sumVderiv = sum(norm(model.recip_lattice * G) * abs(pot(G)) for G in G_vectors(basis))
    diameter = norm(model.lattice * 1 ./ T.(basis.fft_size))
    derivative_term = sumVderiv * diameter / T(2) / sqrtΩ

    extent + derivative_term
end


function bound_potential_perpb1_Vnloc(hamblock, hamblock_Ecut2)
    basis2 = hamblock_Ecut2.basis
    kpoint2 = hamblock_Ecut2.kpoint
    model = basis2.model
    Ecut2 = basis2.Ecut
    kpoint = hamblock.kpoint
    T = eltype(basis2)
    recip_lattice = model.recip_lattice
    @assert any(t isa AtomicLocal for t in model.term_types)
    @assert kpoint.coordinate == kpoint2.coordinate

    # Tabulated values for the L∞ norm of the spherical harmonics Ylm
    Ylm∞ = (sqrt(1 / 4T(π)), sqrt(3 / 4T(π)))  # My guess: (5 / 4π, 7 / 4π)

    # Norms of the G vectors in the Ecut2 basis, ignoring the DC
    Gnorms_Ecut2 = [norm(model.recip_lattice * (G + kpoint.coordinate))
                    for G in G_vectors(kpoint2)]

    # Norms of the G vectors only in Ecut2 basis, but not in Ecut basis
    Gs_complement = setdiff(G_vectors(kpoint2), G_vectors(kpoint))
    Gnorms_complement = [norm(model.recip_lattice * (G + kpoint.coordinate))
                         for G in Gs_complement]

    function norm_projectors(psp, i, l, Nli, Gnorms)
        psp_radials = eval_psp_projection_radial.(psp, i, l, Gnorms)
        sqrt(Nli + sum(abs2, psp_radials)) * Ylm∞[l + 1] / sqrt(model.unit_cell_volume)
    end

    function projector_bound(psp, l)
        @assert psp isa PspHgh
        n_proj_l = size(psp.h[l + 1], 1)

        sum_proj_Ecut2 = zeros(T, n_proj_l)
        sum_proj_complement = zeros(T, n_proj_l)

        for i in 1:n_proj_l
            # Bound for Σ_{G, |G| > qcut} Q^2(t) exp(-t^2) where t = psp.rp[l+1] * |G|.
            Qnloc = DFTK.psp_projection_radial_polynomial(T, psp, i, l)
            qcut = sqrt(2Ecut2) - norm(recip_lattice * kpoint2.coordinate)
            Nli = bound_sum_polynomial_gaussian(Qnloc * Qnloc, recip_lattice,
                                                qcut, β=psp.rp[l + 1])

            sum_proj_Ecut2[i] = norm_projectors(psp, i, l, Nli, Gnorms_Ecut2)
            sum_proj_complement[i] = norm_projectors(psp, i, l, Nli, Gnorms_complement)
        end

        sum_proj_Ecut2' * abs.(psp.h[l + 1]) * sum_proj_complement
    end

    @assert all(element isa ElementPsp for (element, _) in model.atoms)
    sum(maximum(projector_bound.(element.psp, 0:element.psp.lmax)) * length(positions)
        for (element, positions) in model.atoms)
end


@doc raw"""
Estimates a (rough) upper bound for ``\|V φ\|`` for any ``φ`` outside `Ecut`
and `V` are the potential terms of the hamiltonian `hamblock` given
on the `Ecut2` basis.

It is assumed that potential terms are decaying functions in Fourier
space beyond `Ecut2`.

hamblock:        Hamiltonian k-block in Ecut basis
hamblock_Ecut2:  Hamiltonian k-block in Ecut2 basis
"""
function bound_potential_perpb1(hamblock, hamblock_Ecut2)
    @assert hamblock_Ecut2.basis.Ecut > hamblock.basis.Ecut
    @assert hamblock_Ecut2.kpoint.coordinate == hamblock.kpoint.coordinate
    @assert hamblock.basis.model == hamblock_Ecut2.basis.model
    basis2 = hamblock_Ecut2.basis

    if is_cohen_bergstresser(basis2.model)
        @assert hamblock_Ecut2.basis.Ecut ≥ minimal_Ecut2(hamblock.basis)
        (AtomicLocal=bound_potential_perpb1_Vloc(hamblock, hamblock_Ecut2), )
    elseif is_linear_atomic(basis2.model)
        (AtomicLocal=bound_potential_perpb1_Vloc(hamblock, hamblock_Ecut2),
         AtomicNonlocal=bound_potential_perpb1_Vnloc(hamblock, hamblock_Ecut2))
    else
        error("Not implemented")
    end
end

# TODO new
function bound_opnorm_b1_b1perp(Hk, Hk_Ecut2)
    sum(bound_potential_perpb1(Hk, Hk_Ecut2))
end

# TODO new
function bound_opnorm_b1perp_b1perp(Hk, Hk_Ecut2)
    bound_opnorm_b1_b1perp(Hk, Hk_Ecut2)
end
