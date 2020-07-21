# Very basic setup, useful for testing
using DFTK
using PyPlot
using FFTW

BLAS.set_num_threads(4)
FFTW.set_num_threads(4)

include("perturbations.jl")
include("self_energy.jl")

# setting up the model on the fine grid
a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms1 = [Si => [ones(3)/8, -ones(3)/8]]
depl = [0.1, 0, 0]
atoms2 = [Si => [ones(3)/8 + depl, -ones(3)/8]]
model1 = model_atomic(lattice, atoms1, n_electrons=2)
model2 = model_atomic(lattice, atoms2, n_electrons=2)
kgrid = [1, 1, 1]  # k-point grid (Regular Monkhorst-Pack grid)
kcoords1, ksymops1 = bzmesh_ir_wedge(kgrid, model1.symops)
kcoords2, ksymops2 = bzmesh_ir_wedge(kgrid, model2.symops)
terms = ["Kinetic", "AtomicLocal", "AtomicNonlocal"]

Ecut = 50          # kinetic energy cutoff in Hartree
basis1 = PlaneWaveBasis(model1, Ecut; kgrid=kgrid)
basis2 = PlaneWaveBasis(model2, Ecut; kgrid=kgrid)

# parameters
tol = 1e-10
avg = true

# reference computation
println("---------------------")
println("Reference computation")
scfres1 = self_consistent_field(basis1, tol=tol)
λ_ref1 = scfres1.eigenvalues[1][1]
scfres2 = self_consistent_field(basis2, tol=tol)
λ_ref2 = scfres2.eigenvalues[1][1]
diffλ_ref = λ_ref2 - λ_ref1

Ecuts_list = 6:2:18

err_coarse = []
err_fine = []
err_pert2 = []
err_pert3 = []
err_se = []
err_see = []

for Ecuts in Ecuts_list

    println("---------------------")
    println("---------------------")
    println("Ecuts = $(Ecuts)")

    # setting up the coarse grid
    println("---------------------")
    println("setting coarse grid model")
    Ecutf = 2.5 * Ecuts
    basis_coarse1 = PlaneWaveBasis(model1, Ecuts; kgrid=kgrid)
    basis_coarse2 = PlaneWaveBasis(model2, Ecuts; kgrid=kgrid)
    basis_fine1 = PlaneWaveBasis(model1, Ecutf; kgrid=kgrid)
    basis_fine2 = PlaneWaveBasis(model2, Ecutf; kgrid=kgrid)
    nrj1, ham1 = energy_hamiltonian(basis_fine1, nothing, scfres1.occupation)
    nrj2, ham2 = energy_hamiltonian(basis_fine2, nothing, scfres2.occupation)

    # interpolation between coarse and fine grid
    H1 = Array(ham1.blocks[1])
    H2 = Array(ham2.blocks[1])
    println("dense hamiltonian built")
    idcs1, idcs_cmplt1 = DFTK.grid_interpolation_indices(basis_coarse1, basis_fine1)
    idcs2, idcs_cmplt2 = DFTK.grid_interpolation_indices(basis_coarse2, basis_fine2)
    Hs1 = H1[idcs1[1], idcs1[1]]
    Hs2 = H2[idcs2[1], idcs2[1]]

    # computation self-energy correction
    Wsl1 = H1[idcs1[1], idcs_cmplt1[1]]
    Wsl2 = H2[idcs2[1], idcs_cmplt2[1]]
    Hl1 = H1[idcs_cmplt1[1], idcs_cmplt1[1]]
    Hl2 = H2[idcs_cmplt2[1], idcs_cmplt2[1]]
    kpt_fine1 = basis_fine1.kpoints[1]
    kpt_fine2 = basis_fine2.kpoints[1]
    kin1 = Diagonal([sum(abs2, basis_fine1.model.recip_lattice * (G + kpt_fine1.coordinate))
                    for G in G_vectors(kpt_fine1)] ./ 2)[idcs_cmplt1[1], idcs_cmplt1[1]]
    kin2 = Diagonal([sum(abs2, basis_fine2.model.recip_lattice * (G + kpt_fine2.coordinate))
                    for G in G_vectors(kpt_fine2)] ./ 2)[idcs_cmplt2[1], idcs_cmplt2[1]]
    avgW1 = Diagonal(compute_avg(basis_fine1, ham1)[1])[idcs_cmplt1[1], idcs_cmplt1[1]]
    avgW2 = Diagonal(compute_avg(basis_fine2, ham2)[1])[idcs_cmplt2[1], idcs_cmplt2[1]]
    Σ01(λ) = Wsl1 * ((λ*I - (kin1 + avgW1)) \ Wsl1')
    Σ02(λ) = Wsl2 * ((λ*I - (kin2 + avgW2)) \ Wsl2')
    function Σ11(λ)
        T = eltype(basis_fine1)
        scratch = (
            ψ_reals=[zeros(complex(T), basis_fine1.fft_size...) for tid = 1:Threads.nthreads()],
            Hψ_reals=[zeros(complex(T), basis_fine1.fft_size...) for tid = 1:Threads.nthreads()]
        )

        ops_no_kin1 = [op for op in ham1.blocks[1].operators
                      if !(op isa DFTK.FourierMultiplication)]
        H_fine_block_no_kin1 = HamiltonianBlock(basis_fine1, kpt_fine1, ops_no_kin1, scratch)
        Wl1 = Array(H_fine_block_no_kin1)[idcs_cmplt1[1], idcs_cmplt1[1]]
        Σ01(λ) + Wsl1 * ((λ*I - (kin1 + avgW1)) \ ( (Wl1 - avgW1) * (λ*I - (kin1 + avgW1)) \ Wsl1'))
    end
    function Σ12(λ)
        T = eltype(basis_fine2)
        scratch = (
            ψ_reals=[zeros(complex(T), basis_fine2.fft_size...) for tid = 1:Threads.nthreads()],
            Hψ_reals=[zeros(complex(T), basis_fine2.fft_size...) for tid = 1:Threads.nthreads()]
        )

        ops_no_kin2 = [op for op in ham2.blocks[1].operators
                      if !(op isa DFTK.FourierMultiplication)]
        H_fine_block_no_kin2 = HamiltonianBlock(basis_fine2, kpt_fine2, ops_no_kin2, scratch)
        Wl2 = Array(H_fine_block_no_kin2)[idcs_cmplt2[1], idcs_cmplt2[1]]
        Σ02(λ) + Wsl2 * ((λ*I - (kin2 + avgW2)) \ ( (Wl2 - avgW2) * (λ*I - (kin2 + avgW2)) \ Wsl2'))
    end

    # diag on the coarse grid
    println("---------------------")
    println("diag on the coarse grid")
    Es1, Vs1 = eigen(Hs1)
    Es2, Vs2 = eigen(Hs2)
    diffλs = Es2[1] - Es1[1]
    push!(err_coarse, abs(diffλs - diffλ_ref))

    # diag on the fine grid
    println("---------------------")
    println("diag on the fine grid")
    E1, V1 = eigen(H1)
    E2, V2 = eigen(H2)
    diffλ_fine = E2[1] - E1[1]
    push!(err_fine, abs(diffλ_fine - diffλ_ref))

    # perturbation
    println("---------------------")
    println("Perturbation")
    scfres_coarse1 = self_consistent_field(basis_coarse1, tol=tol, callback=info->nothing)
    scfres_coarse2 = self_consistent_field(basis_coarse2, tol=tol, callback=info->nothing)
    Ep_fine, ψp_fine, ρp_fine, egvalp21, egvalp31, egvalp_rr, forcesp_fine, Ep2_fine = perturbation(basis_coarse1, kcoords1, ksymops1, scfres_coarse1, 2.5*Ecuts)
    Ep_fine, ψp_fine, ρp_fine, egvalp22, egvalp32, egvalp_rr, forcesp_fine, Ep2_fine = perturbation(basis_coarse2, kcoords2, ksymops2, scfres_coarse2, 2.5*Ecuts)
    diffλ_pert2 = egvalp22[1][1] - egvalp21[1][1]
    diffλ_pert3 = egvalp32[1][1] - egvalp31[1][1]
    push!(err_pert2, abs(diffλ_pert2 - diffλ_ref))
    push!(err_pert3, abs(diffλ_pert3 - diffλ_ref))

    # diagonalization of H + Σ0(λ)
    println("---------------------")
    println("diagonalization of H + Σ0(λ)")
    Esp1, Vsp1 = eigen( Hs1 + Σ01(Es1[1]) )
    Esp2, Vsp2 = eigen( Hs2 + Σ02(Es2[1]) )
    diffλ_se = Esp2[1] - Esp1[1]
    push!(err_se, abs(diffλ_se - diffλ_ref))

    # diagonalization of H + Σ1(λ)
    println("---------------------")
    println("diagonalization of H + Σ1(λ)")
    Esp1, Vsp1 = eigen( Hs1 + Σ11(Es1[1]) )
    Esp2, Vsp2 = eigen( Hs2 + Σ12(Es2[1]) )
    diffλ_se = Esp2[1] - Esp1[1]
    push!(err_see, abs(diffλ_se - diffλ_ref))

end

# plotting
figure()
semilogy(Ecuts_list, err_coarse, "x-", label="coarse diag")
semilogy(Ecuts_list, err_fine, "x-", label="fine diag")
semilogy(Ecuts_list, err_pert2, "x:", label="pert2")
semilogy(Ecuts_list, err_pert3, "x:", label="pert3")
semilogy(Ecuts_list, err_se, "x--", label="se")
semilogy(Ecuts_list, err_see, "x--", label="see")
ylabel("error on the difference λ2 - λ1")
xlabel("Ecut")
legend()
