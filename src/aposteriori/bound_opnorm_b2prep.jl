
###
### Bounds for \|P_{B_2}^\perp V P_{B_1} \|  (potential_perpb2_b1)
###

@doc raw"""
Estimates an upper bound for ``\|P_{B_2}^\perp Vloc P_{B_1} \|_{op}``

hamblock:  Hamiltonian k-block in small Ecut basis
Ecut2:     Large ecut used for ``B_2``
"""
function bound_potential_perpb2_b1_Vloc(hamblock, Ecut2)
    # Code duplication with bound_perpb2_residual_Vloc

    basis = hamblock.basis
    kpoint = hamblock.kpoint
    @assert any(t isa AtomicLocal for t in basis.model.term_types)
    T = eltype(basis)
    recip_lattice = basis.model.recip_lattice

    accu = zero(T)
    for (element, positions) in basis.model.atoms
        element isa ElementCohenBergstresser && continue
        @assert element isa ElementPsp
        @assert element.psp isa PspHgh
        psp = element.psp

        # We need a bound for Σ_{G, |G| > qcut} Q^2(t) exp(-t^2) / t^4
        # where t = psp.rloc * |G|.
        Qloc = DFTK.psp_local_polynomial(T, psp)
        R(q) = bound_sum_polynomial_gaussian(Qloc * Qloc, recip_lattice, q, β=psp.rloc, k=-4)

        for (i, G) in enumerate(G_vectors(kpoint))
            qcut = sqrt(2Ecut2) - norm(recip_lattice * G)
            accu += R(qcut) * length(positions) / basis.model.unit_cell_volume^2
        end
    end
    sqrt(accu)
end

@doc raw"""
Estimates an upper bound for ``\|P_{B_2}^\perp Vnlock P_{B_1} \|_{op}``

hamblock:  Hamiltonian k-block in small Ecut basis
Ecut2:     Large ecut used for ``B_2``
"""
function bound_potential_perpb2_b1_Vnlock(hamblock, Ecut2)
    # Highly related to bound_potential_perpb1_Vnloc, a lot of code duplication
    basis = hamblock.basis
    kpoint = hamblock.kpoint
    model = basis.model
    kpoint = hamblock.kpoint
    T = eltype(basis)
    recip_lattice = model.recip_lattice
    @assert any(t isa AtomicLocal for t in model.term_types)

    # Tabulated values for the L∞ norm of the spherical harmonics Ylm
    Ylm∞ = (sqrt(1 / 4T(π)), sqrt(3 / 4T(π)))  # My guess: (5 / 4π, 7 / 4π)

    # Norms of the G vectors in the Ecut basis (B1)
    Gnorms = [norm(model.recip_lattice * (G + kpoint.coordinate))
              for G in G_vectors(kpoint)]

    function projector_bound(psp, l)
        @assert psp isa PspHgh
        n_proj_l = size(psp.h[l + 1], 1)

        sum_proj_B1 = zeros(T, n_proj_l)
        sum_proj_B2perp = zeros(T, n_proj_l)

        for i in 1:n_proj_l
            # Bound for Σ_{G, |G| > qcut} Q^2(t) exp(-t^2) where t = psp.rp[l+1] * |G|.
            Qnloc = DFTK.psp_projection_radial_polynomial(T, psp, i, l)
            qcut = sqrt(2Ecut2) - norm(recip_lattice * kpoint.coordinate)
            Nli = bound_sum_polynomial_gaussian(Qnloc * Qnloc, recip_lattice,
                                                qcut, β=psp.rp[l + 1])

            psp_radials = eval_psp_projection_radial.(psp, i, l, Gnorms)
            sum_proj_B1[i] = norm(psp_radials) * Ylm∞[l + 1] / sqrt(model.unit_cell_volume)
            sum_proj_B2perp[i] = sqrt(Nli) * Ylm∞[l + 1] / sqrt(model.unit_cell_volume)
        end

        ret = sum_proj_B2perp' * abs.(psp.h[l + 1]) * sum_proj_B1
        @assert ndims(ret) == 0
        ret
    end

    @assert all(element isa ElementPsp for (element, _) in model.atoms)
    sum(maximum(projector_bound.(element.psp, 0:element.psp.lmax)) * length(positions)
        for (element, positions) in model.atoms)
end

@doc raw"""
Estimates an upper bound for ``\|P_{B_2}^\perp H P_{B_1} \|_{op}``

hamblock:  Hamiltonian k-block in small Ecut basis
Ecut2:     Large ecut used for ``B_2``
"""
function bound_ham_perpb2_b1(hamblock, Ecut2)
    T = eltype(hamblock.basis)
    model = hamblock.basis.model
    if is_cohen_bergstresser(model)
        @assert Ecut2 ≥ minimal_Ecut2(hamblock.basis)
        return (AtomicLocal=zero(T), )
    elseif is_linear_atomic(model)
        (AtomicLocal=bound_potential_perpb2_b1_Vloc(hamblock, Ecut2),
         AtomicNonlocal=bound_potential_perpb2_b1_Vnlock(hamblock, Ecut2))
    else
        error("Not implemented")
    end
end



# TODO new
function bound_opnorm_b1_b2perp(Hk, Hk_Ecut2)
    sum(bound_ham_perpb2_b1(Hk, Hk_Ecut2))
end
