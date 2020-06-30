function total_potential(ρ)
    _, ham = energy_hamiltonian(ρ.basis, nothing, nothing;
                                ρ=ρ, eigenvalues=nothing, εF=nothing)
    total_local_potential(ham)
end
@timing function potential_mixing(basis::PlaneWaveBasis;
                                  n_bands=default_n_bands(basis.model),
                                  ρ=guess_density(basis),
                                  ψ=nothing,
                                  tol=1e-6,
                                  maxiter=100,
                                  solver=scf_nlsolve_solver(),
                                  eigensolver=lobpcg_hyper,
                                  n_ep_extra=3,
                                  determine_diagtol=ScfDiagtol(),
                                  mixing=SimpleMixing(),
                                  is_converged=ScfConvergenceEnergy(tol),
                                  callback=ScfDefaultCallback(),
                                  compute_consistent_energies=true,
                                  )
    T = eltype(basis)
    model = basis.model

    # All these variables will get updated by fixpoint_map
    if ψ !== nothing
        @assert length(ψ) == length(basis.kpoints)
        for ik in 1:length(basis.kpoints)
            @assert size(ψ[ik], 2) == n_bands + n_ep_extra
        end
    end
    occupation = nothing
    eigenvalues = nothing
    εF = nothing
    n_iter = 0
    energies = nothing
    ham = nothing

    _, ham = energy_hamiltonian(ρ.basis, nothing, nothing;
                                ρ=ρ, eigenvalues=nothing, εF=nothing)
    V0 = total_local_potential(ham)
    
    V = V0
    Vprev = V

    tol = 1e-6
    dVol = model.unit_cell_volume / prod(basis.fft_size)

    function EρV(V)
        ham_V = hamiltonian_with_total_potential(ham, V)
        res_V = next_density(ham_V; n_bands=n_bands,
                           ψ=ψ, n_ep_extra=3, miniter=1, tol=tol)
        new_E, new_ham = energy_hamiltonian(basis, res_V.ψ, res_V.occupation;
                                            ρ=res_V.ρ, eigenvalues=res_V.eigenvalues, εF=res_V.εF)
        # println(res_V.eigenvalues[1][5] - res_V.eigenvalues[1][4])
        sum(new_E), res_V.ρ, total_local_potential(new_ham)
    end

    Vs = []
    δVs = []
    Eprev = Inf
    for i = 1:40
        E, ρ, GV = EρV(V)
        println("ΔE this step:         = ", E - Eprev)
        Eprev = E
        δV = GV - V

        # generate new direction ΔV from history
        function weight(dV)
            dVr = reshape(dV, basis.fft_size)
            Gsq = [sum(abs2, model.recip_lattice * G) for G in G_vectors(basis)]
            w = (Gsq .+ 1) ./ (Gsq)
            w[1] = 1
            vec(from_fourier(basis, w .* from_real(basis, dVr).fourier).real)
            dV
        end
        ΔV = δV
        if !isempty(Vs)
            mat = hcat(δVs...) .- vec(δV)
            mat = mapslices(weight, mat; dims=[1])
            alphas = -mat \ weight(vec(δV))
            # alphas = -(mat'mat) * mat' * vec(δV)
            for iα = 1:length(Vs)
                ΔV += reshape(alphas[iα] * (Vs[iα] + δVs[iα] - vec(V) - vec(δV)), basis.fft_size)
            end
        end
        push!(Vs, vec(V))
        push!(δVs, vec(δV))

        # actual step
        # ΔV = δV
        new_V = V + ΔV

        new_E, new_ρ, _ = EρV(new_V)

        ΔE = sum(new_E) - sum(E)
        Δρ = (new_ρ - ρ).real

        println("Step $i")
        slope = dVol*dot(δV, Δρ)
        curv = dVol*(-dot(ΔV, Δρ) + dot(Δρ, apply_kernel(basis, from_real(basis, Δρ); ρ=ρ).real))
        curv = abs(curv)
        println("rel curv: ", curv / (dVol*dot(ΔV, ΔV)))

        # E = slope * t + 1/2 curv * t^2
        topt = -slope/curv
        ΔEopt = -1/2*slope^2 / curv

        println("SimpleSCF actual   ΔE = ", ΔE)
        println("SimpleSCF pred     ΔE = ", slope + curv/2)
        # println("Opt       actual      = ", EρV(V + topt*(new_V - V))[1] - E)
        println("Opt       pred     ΔE = ", ΔEopt)
        println("topt                  = ", topt)
        println()

        V = V + topt*(new_V - V)
        # V = V + (new_V - V)
    end
    V
end
