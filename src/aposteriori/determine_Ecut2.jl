# We refer to the plane-wave basis defined by `Ecut` as ``B₁`` and use ``P_{B₁^⟂}``
# (b1perp) to refer to the projector to the complement. For estimating an upper bound to
# the residual norm `\|P_{B₁^⟂} V u\|` for a trial wavefunction `u` we need to consider
# two parts (see section 4 in DOI 10.1039/d0fd00048e):
#      `\|P_{B₁^⟂ ∩ B₂} V u\|` + `\|P_{B₂^⟂} V u\|`
# where ``B₂`` is the plane-wave basis defined by a bigger cutoff `Ecut2`. The first
# term will be computed explicitly and the second term bounded from above. For this
# the Fourier coefficients with wavevector in ``B₂^⟂`` need to decay with known decay
# properties. For HGH pseudopotentials can only be ensured if `Ecut2` is beyond a certain
# minimal value. For more details see section 4 in DOI 10.1039/d0fd00048e

@doc raw"""
Determine the recommended energy cutoff value `Ecut2` for the plane-wave basis
``B₂`` on which the residual is explicitly computed.
"""
function determine_Ecut2(basis::PlaneWaveBasis)
    model = basis.model

    warnterms = (TermExternal, TermHartree, TermXc, PowerNonlinearity)
    if any(term -> any(isa.(Ref(term), warnterms)), basis.terms)
        @warn("A posteriori error estimates are only implemented for a small number of " *
              "models. Other terms will be completely ignored for estimating the error.")
    end

    maximum(determine_Ecut2, basis.terms)
end


# By default a term does not require any particular Ecut2
determine_Ecut2(term::Term) = false

function determine_Ecut2(term::TermAtomicLocal)
    maximum(determine_Ecut2_local(atom, term.basis.Ecut)
            for (atom, positions) in term.basis.model.atoms)
end

function determine_Ecut2(term::TermAtomicNonlocal)
    # Compute the diameter of the first BZ
    diameter = norm(term.basis.model.recip_lattice * Vec3(1, 1, 1))
    @assert diameter ≥ 0

    maximum(determine_Ecut2_nonlocal(atom, term.basis.Ecut, diameter)
            for (atom, positions) in term.basis.model.atoms)
end


"""
For Cohen-Bergstresser models only a few frequencies of the potential in Fourier
space are non-zero. Therefore if `Ecut2` is large enough than the potential is zero
outside the `Ecut2` basis.
"""
function determine_Ecut2_local(element::ElementCohenBergstresser, Ecut)
    # See section 4.1 in DOI 10.1039/d0fd00048e
    Gmax = sqrt(maximum(keys(element.V_sym))) * (2π / element.lattice_constant)
    return basis.Ecut + sqrt(2basis.Ecut) * Gmax + Gmax^2 / 2
end


determine_Ecut2_local(element::ElementPsp, Ecut) = determine_Ecut2_local(element.psp, Ecut)
function determine_Ecut2_nonlocal(element::ElementPsp, Ecut, brillouin_zone_diameter)
    determine_Ecut2_nonlocal(element.psp, Ecut, brillouin_zone_diameter)
end


# GTH pseudopotentials parts exhibit Gaussian decay in Fourier space beyond a certain
# frequency. These function determine an `Ecut2` to ensure the potentials are only
# decaying *outside* ``B₂`` and that the residual beyond `Ecut2` is dominated
# by the explicitly computed part.
function determine_Ecut2_local(psp::PspHgh, Ecut)
    T = eltype(Ecut)
    # Local part: We want sqrt(2 * Ecut2) - norm(G) ≥ qcut, where G in Ecut basis
    qcut = qcut_psp_local(T, psp)
    min_Ecut2 = (qcut + sqrt(2Ecut))^2 / 2

    # TODO The 4 here is an arbitrary fudge factor, which turned out to make
    #      the explicitly computed part of the residual dominate in silicon.
    4min_Ecut2
end


function determine_Ecut2_nonlocal(psp::PspHgh, Ecut, brillouin_zone_diameter)
    T = eltype(Ecut)
    minimal_Ecuts = T[]

    # Nonlocal projectors: We want sqrt(2 * Ecut2) - norm(k) ≥ qcut,
    #                      where G outside of Ecut2 basis
    proj_idcs = unique((i, l) for (i, l, _) in projector_indices(psp))
    for (i, l) in proj_idcs
        qcut = qcut_psp_projection_radial(T, psp, i, l)
        push!(minimal_Ecuts, (qcut + brillouin_zone_diameter)^2 / 2)
    end

    # TODO The 2 here is an arbitrary fudge factor, which turned out to make
    #      the explicitly computed part of the residual dominate in silicon.
    2maximum(minimal_Ecuts)
end
