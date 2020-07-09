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
atoms = [Si => [ones(3)/8, -ones(3)/8]]
model = model_atomic(lattice, atoms, n_electrons=2)
kgrid = [1, 1, 1]  # k-point grid (Regular Monkhorst-Pack grid)
kcoords, ksymops = bzmesh_ir_wedge(kgrid, model.symops)
terms = ["Kinetic", "AtomicLocal", "AtomicNonlocal"]

Ecut = 50          # kinetic energy cutoff in Hartree
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

# parameters
tol = 1e-10
avg = true

# reference computation
println("---------------------")
println("Reference computation")
scfres = self_consistent_field(basis, tol=tol)
nrj_ref = sum([scfres.energies[term] for term in terms])
λ_ref = scfres.eigenvalues[1][1]

Ecuts_list = 10:2:10

err_coarse = []
err_coarse_scf = []
err_pert = []
err_se = []
err_ite_se = []
err_ite_scf = []
err_λ_se = []
err_λ_coarse = []
err_λ_scf = []
err_λ2_pert = []
err_λ3_pert = []

for Ecuts in Ecuts_list

    println("---------------------")
    println("---------------------")
    println("Ecuts = $(Ecuts)")

    global err_coarse, err_coarse_scf, err_pert, err_se, nrj_ref
    global err_ite_se, err_ite_scf
    global err_λ_se, err_λ_scf, err_λ_coarse, err_λ2_pert, err_λ3_pert, λ_ref

    # setting up the coarse grid
    println("---------------------")
    println("setting coarse grid model")
    Ecutf = 2.5 * Ecuts
    basis_small = PlaneWaveBasis(model, Ecuts; kgrid=kgrid)
    basis_fine = PlaneWaveBasis(model, Ecutf; kgrid=kgrid)
    nrj, ham = energy_hamiltonian(basis_fine, nothing, scfres.occupation)

    # interpolation between small and large grid
    H = Array(ham.blocks[1])
    println("dense hamiltonian built")
    idcs, idcs_cmplt = DFTK.grid_interpolation_indices(basis_small, basis_fine)
    Hs = H[idcs[1], idcs[1]]

    # diag on the coarse grid
    println("---------------------")
    println("diag on the coarse grid")
    Es, Vs = eigen(Hs)
    ψs = [Vs[:, 1:4]]
    ρs = compute_density(basis_small, ψs, scfres.occupation)
    nrj_coarse, hams = energy_hamiltonian(basis_small, ψs, scfres.occupation; ρ=ρs)
    push!(err_coarse, abs(sum([nrj_coarse[term] for term in terms]) - nrj_ref))
    push!(err_λ_coarse, abs(Es[1] - λ_ref))

    # scf on the coarse grid
    println("---------------------")
    println("scf on the coarse grid")
    error_list_scf = []
    nrj_list_scf = []
    function plot_callback(info)
        push!(nrj_list_scf, sum(values(info.energies)))
    end
    scfres_small = self_consistent_field(basis_small, tol=tol, callback=plot_callback)
    display(scfres_small.energies)
    display(scfres_small.eigenvalues)
    push!(err_coarse_scf, abs(sum(values(scfres_small.energies)) - nrj_ref))
    error_list_scf = abs.(nrj_list_scf[2:end] - nrj_list_scf[1:end-1])
    push!(err_ite_scf, error_list_scf)
    push!(err_λ_scf, abs(scfres_small.eigenvalues[1][1] - λ_ref))

    # perturbation
    println("---------------------")
    println("Perturbation")
    Ep_fine, ψp_fine, ρp_fine, egvalp2, egvalp3, egvalp_rr, forcesp_fine, Ep2_fine = perturbation(basis_small, kcoords, ksymops, scfres_small, 2.5*Ecuts)
    push!(err_pert, abs(sum(values(Ep_fine)) - nrj_ref))
    push!(err_λ2_pert, abs(egvalp2[1][1] - λ_ref))
    push!(err_λ3_pert, abs(egvalp3[1][1] - λ_ref))

    # self-energy SCF
    println("---------------------")
    println("Self-energy SCF")
    nrj_se, error_list_se, λ = self_energy_SCF(Hs, basis_small, H, idcs, idcs_cmplt, Es[1]; maxiter=100)
    push!(err_se, abs(sum(values(nrj_se)) - nrj_ref))
    push!(err_ite_se, error_list_se)
    push!(err_λ_se, abs(λ - λ_ref))

end

# plotting
figure()
semilogy(Ecuts_list, err_coarse, "x-", label="coarse diag")
semilogy(Ecuts_list, err_coarse_scf, "x-", label="coarse scf")
semilogy(Ecuts_list, err_pert, "x-", label="pert")
semilogy(Ecuts_list, err_se, "x-", label="se")
ylabel("error")
xlabel("Ecut")
legend()

figure()
for i in 1:length(Ecuts_list)
    global err_ite_se, err_ite_scf
    semilogy(1:length(err_ite_se[i][2:end]), err_ite_se[i][2:end], "x-", label="Ecuts = $(Ecuts_list[i])")
    semilogy(1:length(err_ite_scf[i][2:end]), err_ite_scf[i][2:end], "x--", label="Ecuts = $(Ecuts_list[i])")
end
ylabel("error")
xlabel("ite")
legend()

figure()
semilogy(Ecuts_list, err_λ_coarse, "x-", label="λ coarse diag")
semilogy(Ecuts_list, err_λ_scf, "x-", label="λ coarse scf")
semilogy(Ecuts_list, err_λ2_pert, "x-", label="λ2 pert")
semilogy(Ecuts_list, err_λ3_pert, "x-", label="λ3 pert")
semilogy(Ecuts_list, err_λ_se, "x-", label="λ se")
xlabel("Ecut")
ylabel("error")
legend()

