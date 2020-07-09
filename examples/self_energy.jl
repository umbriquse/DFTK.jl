#
# Self-energy routines for perturbation
#

function self_energy_SCF(Hs, basis_small, Σ, λ0; maxiter=10)

    error = 10
    error_list = []
    λ = λ0
    k = 0
    nrj = 0
    ψsp = 0

    while error > tol && k < maxiter
        k += 1
        Esp, Vsp = eigen( Hs + Σ(λ) )
        ψsp = [Vsp[:, 1:4]]
        ρsp = compute_density(basis_small, ψsp, scfres.occupation)
        nrj, hams = energy_hamiltonian(basis_small, ψsp, scfres.occupation; ρ=ρsp)

        if k == 1
            push!(error_list, NaN)
        else
            error = abs(Esp[1] - λ)
            push!(error_list, error)
        end

        λ = Esp[1]
        println("iteration $(k) : error = $(error)")
    end

    λ, ψsp, nrj, error_list
end
