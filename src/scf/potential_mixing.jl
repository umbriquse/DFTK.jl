@timing function potential_mixing(basis::PlaneWaveBasis;
                                  n_bands=default_n_bands(basis.model),
                                  ρ=guess_density(basis),
                                  ρspin=guess_spin_density(basis),
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
    n_spin = basis.model.n_spin_components

    _, ham = energy_hamiltonian(ρ.basis, nothing, nothing; ρ=ρ, ρspin=ρspin)
    V0 = cat(total_local_potential(ham)..., dims=4)

    V = V0
    Vprev = V

    dVol = model.unit_cell_volume / prod(basis.fft_size)

    function EρV(V)
        Vunpack = [@view V[:, :, :, σ] for σ in 1:n_spin]
        ham_V = hamiltonian_with_total_potential(ham, Vunpack)
        res_V = next_density(ham_V; n_bands=n_bands,
                             ψ=ψ, n_ep_extra=3, miniter=1, tol=tol / 10)
        new_E, new_ham = energy_hamiltonian(basis, res_V.ψ, res_V.occupation;
                                            ρ=res_V.ρout, ρspin=res_V.ρ_spin_out,
                                            eigenvalues=res_V.eigenvalues, εF=res_V.εF)
        ψ = res_V.ψ
        # println(res_V.eigenvalues[1][5] - res_V.eigenvalues[1][4])
        new_E.total, res_V.ρout, res_V.ρ_spin_out, total_local_potential(new_ham)
    end

    Vs = []
    δVs = []
    Eprev = Inf
    for i = 1:maxiter
        E, ρ, ρspin, GV = EρV(V)
        GV = cat(GV..., dims=4)
        println("ΔE this step:         = ", E - Eprev)
        if !isnothing(ρspin)
            println("Magnet                  ", sum(ρspin.real) * dVol)
        end
        Eprev = E
        δV = GV - V

        # generate new direction ΔV from history
        function weight(dV)  # Precondition with Kerker
            dVr = copy(reshape(dV, basis.fft_size..., n_spin))
            Gsq = [sum(abs2, model.recip_lattice * G) for G in G_vectors(basis)]
            w = (Gsq .+ 1) ./ (Gsq)
            w[1] = 1
            # for σ in 1:n_spin
            #     dVr[:, :, :, σ] = from_fourier(basis, w .* from_real(basis, dVr[:, :, :, σ]).fourier).real
            # end
            dV
        end
        ΔV = δV
        if !isempty(Vs)
            mat = hcat(δVs...) .- vec(δV)
            mat = mapslices(weight, mat; dims=[1])
            alphas = -mat \ weight(vec(δV))
            # alphas = -(mat'mat) * mat' * vec(δV)
            for iα = 1:length(Vs)
                ΔV += reshape(alphas[iα] * (Vs[iα] + δVs[iα] - vec(V) - vec(δV)), basis.fft_size..., n_spin)
            end
        end
        push!(Vs, vec(V))
        push!(δVs, vec(δV))

        # actual step
        # ΔV = δV
        new_V = V + ΔV

        new_E, new_ρ, new_ρspin, _ = EρV(new_V)

        ΔE = new_E - E
        abs(ΔE) < tol && break
        Δρ = (new_ρ - ρ).real
        if !isnothing(ρspin)
            Δρspin = (new_ρspin - ρspin).real
            Δρ_RFA     = from_real(basis, Δρ)
            Δρspin_RFA = from_real(basis, Δρspin)

            Δρα = (Δρ + Δρspin) / 2
            Δρβ = (Δρ - Δρspin) / 2
            Δρ = cat(Δρα, Δρβ, dims=4)
        else
            Δρ_RFA = from_real(basis, Δρ)
            Δρspin_RFA = nothing
            Δρ = reshape(Δρ, size(Δρ)..., 1)
        end

        println("Step $i")
        slope = dVol * dot(δV, Δρ)
        KΔρ = apply_kernel(basis, Δρ_RFA, Δρspin_RFA; ρ=ρ, ρspin=ρspin)
        if n_spin == 1
            KΔρ = reshape(KΔρ[1].real, size(KΔρ[1].real)..., 1)
        else
            KΔρ = cat(KΔρ[1].real, KΔρ[2].real, dims=4)
        end

        curv = dVol*(-dot(ΔV, Δρ) + dot(Δρ, KΔρ))
        println("rel curv: ", curv / (dVol*dot(ΔV, ΔV)))
        curv = abs(curv)

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

    Vunpack = [@view V[:, :, :, σ] for σ in 1:n_spin]
    ham = hamiltonian_with_total_potential(ham, Vunpack)
    info = (ham=ham, basis=basis, energies=energies, converged=converged,
            ρ=ρ, ρspin=ρspin, eigenvalues=eigenvalues, occupation=occupation, εF=εF,
            n_iter=n_iter, n_ep_extra=n_ep_extra, ψ=ψ)
    info
end
