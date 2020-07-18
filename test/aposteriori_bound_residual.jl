using Test
using DFTK
using DFTK: bound_residual_b2perp, determine_Ecut2, diagonalize_all_kblocks
include("testcases.jl")

function aposteriori_hcore(system=magnesium; n_bands=8, tol=1e-6, Ecut=10)
    Mg = ElementPsp(system.atnum, psp=load_psp(system.psp))
    atoms = [Mg => system.positions]
    model = model_atomic(system.lattice, atoms)
    basis = PlaneWaveBasis(model, Ecut, kgrid=[1, 1, 1])

    # Diagonalise on small basis
    ham = Hamiltonian(basis)
    eigres = diagonalize_all_kblocks(DFTK.lobpcg_hyper, ham, n_bands; tol=tol)

    # Build big basis
    Ecut2 = DFTK.determine_Ecut2(basis)
    @test Ecut2 ≈ 76.81033067937457
    basis_Ecut2 = PlaneWaveBasis(basis, Ecut2)
    X_Ecut2 = DFTK.interpolate_blochwave(eigres.X, basis, basis_Ecut2)
    ham_Ecut2 = Hamiltonian(basis_Ecut2)

    # residual_norm_terms = [[bound_perpb2_residual(ham.blocks[ik], ψnk, Ecut2)
    #                         for ψnk in eachcol(data.X[ik])]
    #                        for ik in 1:length(basis.kpoints)]

    merge(eigres, (ham=ham, X_Ecut2=X_Ecut2, ham_Ecut2=ham_Ecut2))
end


@testset "bound_residual_b2perp magnesium" begin
    data = aposteriori_hcore()
    ham, ham_Ecut2 = data.ham, data.ham_Ecut2
    ik = 1  # Data only has the Gamma point

    ref_Mg_local = [3.9636529610451866e-8, 2.3907303706544993e-8, 6.187742375108702e-8,
                    1.0385062588623786e-7, 1.0385077842967404e-7, 1.25985029914099e-7,
                    1.259850467588589e-7, 1.6584414555383994e-7]
    ref_Mg_nonlocal = [2.102972407180026e-9, 1.6695241819563645e-9, 2.475192569891374e-9,
                       2.4545111391724337e-9, 2.4545111350264183e-9, 1.3067109627636064e-9,
                       1.3067103303097969e-9, 3.459871156605743e-10]

    cases = [(DFTK.TermAtomicLocal, DFTK.RealSpaceMultiplication, ref_Mg_local),
             (DFTK.TermAtomicNonlocal, DFTK.NonlocalOperator, ref_Mg_nonlocal),  ]
    for (termT, opT, ref) in cases
        tidx = findfirst(t -> t isa termT, ham.basis.terms)
        @test ham_Ecut2.basis.terms[tidx] isa termT
        @test ham.blocks[ik].operators[tidx] isa opT
        @test ham_Ecut2.blocks[ik].operators[tidx] isa opT

        res = bound_residual_b2perp(ham.basis.terms[tidx],
                                    ham_Ecut2.basis.terms[tidx],
                                    ham.blocks[ik].operators[tidx],
                                    ham_Ecut2.blocks[ik].operators[tidx],
                                    data.X[ik])
        @test norm(res - ref) < 1e-12
    end

    totalres = bound_residual_b2perp(ham.blocks[ik], ham_Ecut2.blocks[ik], data.X[ik])
    @test totalres ≈ ref_Mg_local + ref_Mg_nonlocal
end
