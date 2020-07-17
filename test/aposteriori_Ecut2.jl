using Test
using DFTK: determine_Ecut2
using DFTK

@testset "Ecut2 Silicon Cohen-Bergstresser" begin
    function make_basis(Ecut)
        Si = ElementCohenBergstresser(:Si)
        a = Si.lattice_constant
        lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
        atoms = [Si => [ones(3)/8, -ones(3)/8]]
        model = Model(lattice; atoms=atoms, terms=[Kinetic(), AtomicLocal()])
        PlaneWaveBasis(model, Ecut, kgrid=[1, 1, 1])
    end

    @test determine_Ecut2(make_basis( 10)) ≥  21
    @test determine_Ecut2(make_basis( 20)) ≥  34.5
    @test determine_Ecut2(make_basis( 30)) ≥  47.5
    @test determine_Ecut2(make_basis(100)) ≥ 130.5
end


@testset "Ecut2 linear Silicon" begin
    function make_basis(Ecut)
        a = 10.263141334305942  # Silicon lattice constant in Bohr
        lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
        Si = ElementPsp(:Si, psp=load_psp("hgh/lda/si-q4"))
        atoms = [Si => [ones(3)/8, -ones(3)/8]]
        model = model_atomic(lattice, atoms)
        PlaneWaveBasis(model, Ecut, kgrid=[1, 1, 1])
    end

    @test determine_Ecut2(make_basis( 10)) ≥  40
    @test determine_Ecut2(make_basis( 20)) ≥  80
    @test determine_Ecut2(make_basis( 30)) ≥ 120
    @test determine_Ecut2(make_basis(100)) ≥ 400
end
