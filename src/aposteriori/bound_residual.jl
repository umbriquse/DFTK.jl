
@doc raw"""
Estimates an upper bound for ``\|P_{B₂}^⟂ H_k ψnk\|`` if `ψnk` is an
eigenvector in the small `Ecut` basis. `Hk` and `Hk_Ecut2` is the same
``k``-block Hamiltonian in the ``B₁`` and ``B₂`` basis respectively.
"""
function bound_residual_b2perp(Hk, Hk_Ecut2, ψk::AbstractVecOrMat)
    @assert length(Hk.basis.terms) == length(Hk_Ecut2.basis.terms)
    @assert length(Hk.operators) == length(Hk_Ecut2.operators)
    n_bands = size(ψk, 2)
    n_terms = length(Hk.basis.terms)
    basis = Hk.basis
    basis_Ecut2 = Hk_Ecut2.basis

    res = similar(real(eltype(ψk)), n_terms)
    res .= 0
    for iterm in 1:n_terms
        res .+= bound_residual_b2perp(basis.terms[iterm],
                                      basis_Ecut2.terms[iterm],
                                      Hk.operators[iterm],
                                      Hk_Ecut2.operators[iterm], ψk)
    end

    res
end


# Fallback implementation: Just returns array of falses
function bound_residual_b2perp(term::Term, term_Ecut2::Term,
                               operator::RealFourierOperator,
                               operator_Ecut2::RealFourierOperator,
                               ψk::AbstractVecOrMat)
    similar(ψk, Bool, size(ψk, 2))
end


function bound_residual_b2perp(term::TermAtomicLocal,
                               term_Ecut2::TermAtomicLocal,
                               operator::RealSpaceMultiplication,
                               operator_Ecut2::RealSpaceMultiplication,
                               ψk::AbstractMatrix)
    # TODO Integrate this with the function below
    [bound_residual_b2perp(term, term_Ecut2, operator, operator_Ecut2, ψnk)
     for ψnk in eachcol(ψk)]
end
function bound_residual_b2perp(term::TermAtomicLocal,
                               term_Ecut2::TermAtomicLocal,
                               operator::RealSpaceMultiplication,
                               operator_Ecut2::RealSpaceMultiplication,
                               ψnk::AbstractVector; s=0)
    # s is the Sobolev exponent for the norm estimate. 0 turned out to work best
    basis  = term.basis
    Vlock  = operator
    kpoint = operator.kpoint
    Ecut2  = term_Ecut2.basis.Ecut
    T = eltype(basis)
    recip_lattice = basis.model.recip_lattice
    @assert length(ψnk) == length(G_vectors(kpoint))


    accu = zero(T)
    for (element, positions) in basis.model.atoms
        element isa ElementCohenBergstresser && continue  # TODO generalise
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
            factor = s == 0 ? one(T) : (1 + Gnorm^2)^s
            accu += factor * abs2(ψnk[i]) * R(qcut) * length(positions)
        end
    end

    if s == 0
        prefac = length(ψnk)
    else
        prefac = sum((1 + norm(recip_lattice * G)^2)^(-s) for G in G_vectors(kpoint))
    end
    sqrt(prefac * accu / basis.model.unit_cell_volume^2)
end


function bound_residual_b2perp(term::TermAtomicNonlocal,
                               term_Ecut2::TermAtomicNonlocal,
                               operator::NonlocalOperator,
                               operator_Ecut2::NonlocalOperator,
                               ψk::AbstractMatrix)
    # TODO Integrate this with the function below
    [bound_residual_b2perp(term, term_Ecut2, operator, operator_Ecut2, ψnk)
     for ψnk in eachcol(ψk)]
end
function bound_residual_b2perp(term::TermAtomicNonlocal,
                               term_Ecut2::TermAtomicNonlocal,
                               operator::NonlocalOperator,
                               operator_Ecut2::NonlocalOperator,
                               ψnk::AbstractVector)
    basis  = term.basis
    Vlock  = operator
    kpoint = operator.kpoint
    Ecut2  = term_Ecut2.basis.Ecut
    T = eltype(basis)
    recip_lattice = basis.model.recip_lattice

    @assert length(ψnk) == length(G_vectors(kpoint))
    fnorm = norm(Vlock.D * (Vlock.P' * ψnk))

    accu = zero(T)
    for (element, positions) in basis.model.atoms
        element isa ElementCohenBergstresser && continue  # TODO generalise
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
