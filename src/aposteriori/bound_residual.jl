
###
### Bounds for \|P_{B_2}^\perp V ψnk\|  (perpb2_residual)
###

@doc raw"""
Estimates an upper bound for ``\|P_{B_2}^\perp V_loc ψnk\|`` using varying
Sobolev exponents `s`.

hamblock:  Hamiltonian k-block in small Ecut basis
ψnk:       Eigenvector for a particular band and k-point on small ecut
Ecut2:     Large ecut used for ``B_2``
s:         Sobolev exponent
"""
function bound_perpb2_residual_Vloc_sobolev(hamblock, ψnk::AbstractVector, Ecut2; s=2)
    basis = hamblock.basis
    kpoint = hamblock.kpoint
    @assert length(ψnk) == length(G_vectors(kpoint))
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
            Gnorm = norm(recip_lattice * G)
            qcut = sqrt(2Ecut2) - Gnorm
            accu += (1 + Gnorm^2)^s * abs2(ψnk[i]) * R(qcut) * length(positions)
        end
    end

    prefac = sum((1 + norm(recip_lattice * G)^2)^(-s) for G in G_vectors(kpoint))
    sqrt(prefac * accu / basis.model.unit_cell_volume^2)
end



@doc raw"""
Estimates an upper bound for ``\|P_{B_2}^\perp V_loc ψnk\|``

hamblock:  Hamiltonian k-block in small Ecut basis
ψnk:     Eigenvector for a particular band and k-point on small ecut
Ecut2:   Large ecut used for ``B_2``
"""
function bound_perpb2_residual_Vloc(hamblock, ψnk::AbstractVector, Ecut2; α=1)
    basis = hamblock.basis
    kpoint = hamblock.kpoint
    @assert length(ψnk) == length(G_vectors(kpoint))
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
            factor = α == 0 ? one(T) : abs(ψnk[i])^2α
            qcut = sqrt(2Ecut2) - norm(recip_lattice * G)
            accu += factor * R(qcut) * length(positions)
        end
    end

    if α == 0
        prefac = one(T)
    elseif α == 1
        prefac = length(ψnk)
    else
        prefac = sum(@. abs(ψnk)^(2-2α))
    end
    sqrt(prefac * accu) / basis.model.unit_cell_volume
end

@doc raw"""
Estimates an upper bound for ``\|P_{B_2}^\perp V_nlock ψnk\|``

hamblock:  Hamiltonian k-block in small Ecut basis
ψnk:       Eigenvector for a particular band and k-point on small Ecut
Ecut2:     Large ecut used for ``B_2``
"""
function bound_perpb2_residual_Vnlock(hamblock, ψnk::AbstractVector, Ecut2)
    basis = hamblock.basis
    kpoint = hamblock.kpoint
    T = eltype(basis)
    recip_lattice = basis.model.recip_lattice
    @assert length(ψnk) == length(G_vectors(kpoint))

    # Compute norm of vector f
    idx_nonlocal = only(findall(t -> t isa AtomicNonlocal, basis.model.term_types))
    Vlock = hamblock.operators[idx_nonlocal]
    fnorm = norm(Vlock.D * (Vlock.P' * ψnk))

    accu = zero(T)
    for (element, positions) in basis.model.atoms
        @assert element isa ElementPsp
        @assert element.psp isa PspHgh
        psp = element.psp

        # Ignore m ... sum over it done implicitly by the (2l+1)
        proj_idcs = unique((i, l) for (i, l, _) in DFTK.projector_indices(psp))
        for (i, l) in proj_idcs
            # Bound for Σ_{G, |G| > qcut} Q^2(t) exp(-t^2) where t = psp.rp[l+1] * |G|.
            Qnloc = DFTK.psp_projection_radial_polynomial(T, psp, i, l)
            qcut = sqrt(2Ecut2) - norm(recip_lattice * kpoint.coordinate)
            N = bound_sum_polynomial_gaussian(Qnloc * Qnloc, recip_lattice,
                                              qcut, β=psp.rp[l + 1])

            accu += (2l + 1) * length(positions) * N
        end
    end

    fnorm * sqrt(accu / 4T(π) / basis.model.unit_cell_volume)
end


@doc raw"""
Estimates an upper bound for ``\|P_{B_2}^\perp H ψnk\|`` if ``ψnk`` is given
on the `Ecut` basis.

hamblock:  Hamiltonian k-block in small `Ecut` basis
ψnk:       Eigenvector for a particular band and k-point on small `Ecut`
Ecut2:     Large ecut used for ``B_2``
"""
function bound_perpb2_residual(hamblock, ψnk::AbstractVector, Ecut2)
    T = eltype(hamblock.basis)
    model = hamblock.basis.model
    if is_cohen_bergstresser(model)
        @assert Ecut2 ≥ minimal_Ecut2(hamblock.basis)
        return (Kinetic=zero(T), AtomicLocal=zero(T))
    elseif is_linear_atomic(model)
        (Kinetic=zero(T),
         AtomicLocal=bound_perpb2_residual_Vloc(hamblock, ψnk, Ecut2),
         AtomicNonlocal=bound_perpb2_residual_Vnlock(hamblock, ψnk, Ecut2),
         Ewald=zero(T),
         PspCorrection=zero(T)
        )
    else
        error("Not implemented")
    end
end


# TODO new
function bound_residual_b2perp(Hk, Hk_Ecut2, ψk_Ecut2)
    [bound_perpb2_residual(Hk, @view ψk_Ecut2[:, iband], Hk_Ecut2.basis.Ecut)
     for iband in size(ψk_Ecut2, 2)]
end
