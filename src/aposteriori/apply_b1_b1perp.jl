@doc raw"""
Get P_{B_1^⟂ ∩ B_2} op P_{B_1} ψk_Ecut2 as a dense matrix, where ψk_Ecut2 is a matrix
with columns in the Ecut basis.
"""
function apply_op_perpb1capb2(opblock, opblock_Ecut2, ψk_Ecut2::AbstractMatrix)
    basis2 = opblock_Ecut2.basis
    kpoint2 = opblock_Ecut2.kpoint
    kpoint = opblock.kpoint
    T = eltype(basis2)
    @assert kpoint2.coordinate == kpoint.coordinate
    @assert size(ψk_Ecut2, 1) == length(G_vectors(kpoint2))

    b1_in_b2 = indexin(G_vectors(kpoint), G_vectors(kpoint2))
    not_in_b1 = [i for i in 1:length(G_vectors(kpoint2)) if !(i in b1_in_b2)]
    @assert length(b1_in_b2) + length(not_in_b1) == length(G_vectors(kpoint2))

    # Compute the part inside B2 explicitly and select
    # only the indices inside B_1^⟂ ∩ B_2.
    (opblock_Ecut2 * ψk_Ecut2)[not_in_b1, :]
end

@doc raw"""
Get P_{B_1^⟂ ∩ B_2} H P_{B_1} ψk_Ecut2 as a dense matrix, where ψk_Ecut2 is a matrix
with columns in the Ecut2 basis.
"""
function apply_hamiltonian_perpb1capb2(hamblock, hamblock_Ecut2, ψk_Ecut2::AbstractMatrix)
    @assert hamblock_Ecut2.basis.Ecut > hamblock.basis.Ecut
    @assert hamblock_Ecut2.kpoint.coordinate == hamblock.kpoint.coordinate
    @assert hamblock.basis.model == hamblock_Ecut2.basis.model
    basis2 = hamblock_Ecut2.basis

    idx_local = only(findall(t -> t isa AtomicLocal, basis2.model.term_types))
    Vloc = hamblock.operators[idx_local]
    Vloc_Ecut2 = hamblock_Ecut2.operators[idx_local]

    if is_cohen_bergstresser(basis2.model)
        @assert basis2.Ecut ≥ minimal_Ecut2(hamblock.basis)
        apply_op_perpb1capb2(Vloc, Vloc_Ecut2, ψk_Ecut2)
    elseif is_linear_atomic(basis2.model)
        idx_nonlocal = only(findall(t -> t isa AtomicNonlocal,
                                    basis2.model.term_types))
        Vnlock = hamblock.operators[idx_nonlocal]
        Vnlock_Ecut2 = hamblock_Ecut2.operators[idx_nonlocal]

        (apply_op_perpb1capb2(Vloc, Vloc_Ecut2, ψk_Ecut2)
         + apply_op_perpb1capb2(Vnlock, Vnlock_Ecut2, ψk_Ecut2))
    else
        error("Not implemented")
    end
end


# TODO new
function apply_b1perpcapb2_b1(Hk, Hk_Ecut2, ψk_Ecut2)
    apply_hamiltonian_perpb1capb2(Hk, Hk_Ecut2, ψk_Ecut2)
end
