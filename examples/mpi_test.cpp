// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

/// Minimal MPI+MFEM test with step-by-step diagnostics
#include <mfem.hpp>
#include <HYPRE.h>
#include <iostream>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    HYPRE_Init();  // Required for HYPRE 3.x
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "[" << rank << "] Step 1: MPI OK (" << size << " procs)\n" << std::flush;

    // 1. Serial mesh
    auto serial = mfem::Mesh::MakeCartesian3D(3, 3, 3,
        mfem::Element::TETRAHEDRON, 1.0, 1.0, 1.0);
    std::cout << "[" << rank << "] Step 2: Serial mesh OK (" << serial.GetNE() << " elems)\n" << std::flush;

    // 2. ParMesh
    mfem::ParMesh pmesh(MPI_COMM_WORLD, serial);
    std::cout << "[" << rank << "] Step 3: ParMesh OK (" << pmesh.GetNE() << " local elems)\n" << std::flush;

    // 3. H1 FE space
    mfem::H1_FECollection fec(1, 3);
    mfem::ParFiniteElementSpace fespace(&pmesh, &fec);
    std::cout << "[" << rank << "] Step 4: FESpace OK (local=" << fespace.GetTrueVSize() << ")\n" << std::flush;

    // 4. Assemble
    mfem::ConstantCoefficient one(1.0);
    mfem::ParBilinearForm a(&fespace);
    a.AddDomainIntegrator(new mfem::DiffusionIntegrator(one));
    std::cout << "[" << rank << "] Step 5: Added integrator\n" << std::flush;

    a.Assemble();
    std::cout << "[" << rank << "] Step 6: Assembled\n" << std::flush;

    // 5. Essential BC
    mfem::Array<int> ess_bdr(pmesh.bdr_attributes.Max());
    ess_bdr = 1;
    mfem::Array<int> ess_tdof;
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof);
    std::cout << "[" << rank << "] Step 7: Essential DOFs (" << ess_tdof.Size() << ")\n" << std::flush;

    // 6. FormSystemMatrix
    mfem::OperatorHandle A_handle;
    a.FormSystemMatrix(ess_tdof, A_handle);
    std::cout << "[" << rank << "] Step 8: FormSystemMatrix OK\n" << std::flush;

    auto* A = A_handle.As<mfem::HypreParMatrix>();
    std::cout << "[" << rank << "] Step 9: HypreParMatrix OK (rows=" << A->Height() << ")\n" << std::flush;

    // 7. AMG
    mfem::HypreBoomerAMG amg(*A);
    amg.SetPrintLevel(0);
    std::cout << "[" << rank << "] Step 10: AMG OK\n" << std::flush;

    // 8. CG solve
    mfem::Vector b(fespace.GetTrueVSize());
    mfem::Vector x(fespace.GetTrueVSize());
    b.Randomize(42 + rank);
    x = 0.0;
    for (int i = 0; i < ess_tdof.Size(); i++) b(ess_tdof[i]) = 0.0;

    mfem::CGSolver cg(MPI_COMM_WORLD);
    cg.SetOperator(*A);
    cg.SetPreconditioner(amg);
    cg.SetRelTol(1e-6);
    cg.SetMaxIter(100);
    cg.SetPrintLevel(0);
    std::cout << "[" << rank << "] Step 11: CG setup OK, solving...\n" << std::flush;

    cg.Mult(b, x);
    std::cout << "[" << rank << "] Step 12: CG done! converged=" << cg.GetConverged()
              << " iter=" << cg.GetNumIterations() << "\n" << std::flush;

    MPI_Finalize();
    return 0;
}
