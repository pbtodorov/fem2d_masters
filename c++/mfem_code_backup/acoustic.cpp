//                                MFEM Example 3
//
// Compile with: make ex3
//
// Sample runs:  ex3 -m ../data/star.mesh
//               ex3 -m ../data/beam-tri.mesh -o 2
//               ex3 -m ../data/beam-tet.mesh
//               ex3 -m ../data/beam-hex.mesh
//               ex3 -m ../data/beam-hex.mesh -o 2 -pa
//               ex3 -m ../data/fichera-amr.mesh
//               ex3 -m ../data/ref-prism.mesh -o 1
//               ex3 -m ../data/octahedron.mesh -o 1
//               ex3 -m ../data/star-surf.mesh -o 1
//               ex3 -m ../data/mobius-strip.mesh -f 0.1
//               ex3 -m ../data/klein-bottle.mesh -f 0.1
//
// Device sample runs:
//               ex3 -m ../data/star.mesh -pa -d cuda
//               ex3 -m ../data/star.mesh -pa -d raja-cuda
//               ex3 -m ../data/star.mesh -pa -d raja-omp
//               ex3 -m ../data/beam-hex.mesh -pa -d cuda
//
// Description:  This example code solves a simple electromagnetic diffusion
//               problem corresponding to the second order definite Maxwell
//               equation curl curl E + E = f with boundary condition
//               E x n = <given tangential field>. Here, we use a given exact
//               solution E and compute the corresponding r.h.s. f.
//               We discretize with Nedelec finite elements in 2D or 3D.
//
//               The example demonstrates the use of H(curl) finite element
//               spaces with the curl-curl and the (vector finite element) mass
//               bilinear form, as well as the computation of discretization
//               error when the exact solution is known. Static condensation is
//               also illustrated.
//
//               We recommend viewing examples 1-2 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>


using namespace std;
using namespace mfem;

void printCurrentSol(const mfem::BlockVector& sol, std::ofstream& outFile);
void saveSparseMatrixInFile(const std::string& filename, const SparseMatrix& M);
void saveSparseVectorInFile(LinearForm* b, const std::string& configPath, const std::string& filename)
{
    std::ofstream vectorStream(configPath + "\\MFEM\\" +filename);

    auto data = b->GetData();
    for (int i = 0; i < b->Size(); i++)
    {
        if (data[i] != 0)
        {
            vectorStream << i << ", " << data[i] << '\n';
        }
    }
}

std::vector<std::vector<int>> readElementsCSV(std::string filename) {
    filename += "\\elements.csv";
    std::vector<std::vector<int>> data;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }

    while (std::getline(file, line)) {
        std::vector<int> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));  // convert to double
        }

        data.push_back(row);
    }

    file.close();
    return data;
}

std::vector<std::pair<double, double>> readNodesCSV(std::string filename) {
    filename += "\\nodes.csv";
    std::vector<std::pair<double, double>> data;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }

    while (std::getline(file, line)) {
        std::pair<double, double> row;
        std::stringstream ss(line);
        std::string cell;

        std::getline(ss, cell, ',');
        row.first = std::stod(cell);

        std::getline(ss, cell, ',');
        row.second = std::stod(cell);

        data.push_back(row);
    }

    file.close();
    return data;
}

void AssembleSum(const mfem::BlockMatrix& A, const mfem::BlockMatrix& B, mfem::BlockMatrix& Result);

void AssembleDiff(const mfem::BlockMatrix& A, const mfem::BlockMatrix& B, mfem::BlockMatrix& Result);

bool IsSymmetric(const mfem::SparseMatrix& A, double tol = 1e-10)
{
    for (int i = 0; i < A.NumRows(); ++i)
    {
        for (int j = i; j < A.NumCols(); ++j)  // Only check upper triangle
        {
            double val_ij = A(i, j);
            double val_ji = A(j, i);

            // Check if the difference is within tolerance
            if (std::fabs(val_ij - val_ji) > tol)
            {
                return false;
            }
        }
    }
    return true;
}

void ConvertBlockMatrixToHypreParMatrix(const mfem::BlockMatrix& block_matrix,
    mfem::HypreParMatrix*& hypre_par_matrix, mfem::Mesh& mesh, const FiniteElementSpace& orig, const FiniteElementCollection* fec)
{
    if (mesh.GetNE() == 0) {
        std::cerr << "Error: Empty mesh provided." << std::endl;
        hypre_par_matrix = nullptr;
        return;
    }
    // Get the number of blocks in the BlockMatrix
    int num_blocks_x = block_matrix.Height();
    int num_blocks_y = block_matrix.Width();

    ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
    MPI_Comm comm = pmesh->GetComm();
    ParFiniteElementSpace* fespace = new ParFiniteElementSpace(orig, *pmesh, fec);

    SparseMatrix* global_mat = block_matrix.CreateMonolithic();

    // Get the appropriate row/column partitioning
    HYPRE_Int* row_starts = fespace->GetDofOffsets();
    HYPRE_Int* col_starts = fespace->GetDofOffsets();

    // Create the HypreParMatrix using the global sparse matrix
    hypre_par_matrix = new HypreParMatrix(pmesh->GetComm(),
        fespace->GlobalTrueVSize(),
        fespace->GlobalTrueVSize(),
        row_starts,
        col_starts,
        global_mat);
    if (hypre_par_matrix->Height())
    {
        std::cout << ":)";
    }

    delete global_mat;
    delete fespace;
    delete pmesh;
    if (hypre_par_matrix->Height())
    {
        std::cout << ":)";
    }
}

pair<int, int> SortedEdge(int ind1, int ind2)
{
    if (ind1 > ind2)
    {
        return {ind2, ind1};
    }
    return { ind1, ind2 };
}

// separate here the edges that we want to have free surface
void AddBoundaryFromTriangles(Mesh& mesh, int free_surface_attr)
{
    map<pair<int, int>, vector<int>> edge_to_elements;

    // First pass: collect edge occurrences
    for (int i = 0; i < mesh.GetNE(); i++)
    {
        Element* el = mesh.GetElement(i);
        MFEM_ASSERT(el->GetType() == Element::TRIANGLE, "Not a triangle.");

        const int* v = el->GetVertices();

        auto e1 = SortedEdge(v[0], v[1]);
        auto e2 = SortedEdge(v[1], v[2]);
        auto e3 = SortedEdge(v[2], v[0]);

        edge_to_elements[e1].push_back(i);
        edge_to_elements[e2].push_back(i);
        edge_to_elements[e3].push_back(i);
    }

    // Second pass: add boundary segments for edges used only once
    for (const auto& entry : edge_to_elements)
    {
        const auto& edge = entry.first;
        const auto& tris = entry.second;

        if (tris.size() == 1) // boundary edge
        {
            // Get coordinates of the two vertices
            const auto& v0_coords = mesh.GetVertex(edge.first);
            const auto& v1_coords = mesh.GetVertex(edge.second);

            // Check if both vertices lie on y = 500 (with tolerance to handle FP precision)
            // Assume we will never get to mesh with min edge 1.5 <-> MaxCellMeasure 4
            double tol = 1.5;
            if (std::abs(v0_coords[1] - 500.0) < tol && std::abs(v1_coords[1] - 500.0) < tol)
            {
                Segment* seg = new Segment;
                Array<int> v(2);
                v[0] = edge.first;
                v[1] = edge.second;
                seg->SetVertices(v);
                seg->SetAttribute(free_surface_attr);
                mesh.AddBdrElement(seg);
            }
            else
            {
                Segment* seg = new Segment;
                Array<int> v(2);
                v[0] = edge.first;
                v[1] = edge.second;
                seg->SetVertices(v);
                seg->SetAttribute(1);
                mesh.AddBdrElement(seg);
            }
        }
    }
}

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector&, Vector&);
void f_exact(const Vector&, Vector&);
real_t freq = 1.0, kappa;
int dim;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    // 1. Parse command-line options.
    const char* mesh_file = "../data/beam-tet.mesh";
    int order = 1;
    bool static_cond = false;
    bool pa = false;
    bool nc = false;
    const char* device_config = "cpu";
    bool visualization = 1;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
        "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
        "Finite element order (polynomial degree).");
    args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
        " solution.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
        "--no-static-condensation", "Enable static condensation.");
    args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
        "--no-partial-assembly", "Enable Partial Assembly.");
    args.AddOption(&nc, "-nc", "--non-conforming", "-c",
        "--conforming",
        "Mark the mesh as nonconforming before partitioning.");
    args.AddOption(&device_config, "-d", "--device",
        "Device configuration string, see Device::Configure().");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
        "--no-visualization",
        "Enable or disable GLVis visualization.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);
    kappa = freq * M_PI;

    // 2. Enable hardware devices such as GPUs, and programming models such as
    //    CUDA, OCCA, RAJA and OpenMP based on command line options.
    Device device(device_config);
    device.Print();

    // 3. Read the mesh from the given mesh file. We can handle triangular,
    //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
    //    the same code.
    //Mesh* mesh = new Mesh(mesh_file, 1, 1);
    //dim = mesh->Dimension();
    //int sdim = mesh->SpaceDimension();
    //if (nc)
    //{
    //    // Can set to false to use conformal refinement for simplices.
    //    mesh->EnsureNCMesh(true);
    //}
    //0. Initialize mesh
    std::string configPath = "D:\\Serious\\FMI\\Магистър\\Дипломен семестър\\FEM2D_config";
    std::vector<std::vector<int>> elementsFromCSV = readElementsCSV(configPath);
    std::vector<std::pair<double, double>> nodesFromCSV = readNodesCSV(configPath);

    int num_nodes = nodesFromCSV.size();

    mfem::Mesh mesh(2, nodesFromCSV.size(), elementsFromCSV.size(), 0, 2); // 2D, no boundary yet

    for (const std::pair<double, double>& node : nodesFromCSV)
    {
        double x = node.first;
        double y = node.second;
        mesh.AddVertex(mfem::Vector({ x, y }));
    }
    // Add elements (assume triangular elements)
    for (const std::vector<int>& element : elementsFromCSV)
    {
        int n1 = element[0];
        int n2 = element[1];
        int n3 = element[2];
        mfem::Array<int> nodes{ n1, n2, n3 };
        mesh.AddTriangle(nodes, 1); // 1 is region attribute
    }
    int free_surface_attr = 2;
    AddBoundaryFromTriangles(mesh, free_surface_attr);
    mesh.FinalizeTriMesh(1);
    //Paralel mesh
    //ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, mesh);

    // Save indices of non free surface boudary nodes
    std::set<int> fakeBoundaryNodes;
    for (int i = 0; i < mesh.GetNBE(); i++)
    {
        Element* be = mesh.GetBdrElement(i);
        const int* v = be->GetVertices();
        // not free surface
        if (be->GetAttribute() == 1)
        {
            fakeBoundaryNodes.emplace(v[0]);
            fakeBoundaryNodes.emplace(v[1]);
        }
    }
    std::ofstream nonFreeSurfaceO(configPath + "\\nonFreeSurfaceNodes");
    
    auto it = fakeBoundaryNodes.begin();
    int el = *it;
    nonFreeSurfaceO << el;
    it++;
    auto last = std::prev(fakeBoundaryNodes.end());
    for (; it != last; ++it) {
        el = *it;
        nonFreeSurfaceO << ", " << el;
    }
    return 0;
    int sdim = mesh.SpaceDimension();
    dim = mesh.Dimension();

    // 4. Refine the mesh to increase the resolution. In this example we do
    //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
    //    largest number that gives a final mesh with no more than 50,000
    //    elements.
    {
   /*     int ref_levels =
            (int)floor(log(50000. / mesh.GetNE()) / log(2.) / dim);
        for (int l = 0; l < ref_levels; l++)
        {
            mesh.UniformRefinement();
        }*/
    }

    // 5. Define a finite element space on the mesh. Here we use the H1
    //    finite elements of the specified order.
    FiniteElementCollection* fec = new LinearFECollection();
    int vectorDimentions = 2;
    FiniteElementSpace* fespace = new FiniteElementSpace(&mesh, fec, vectorDimentions);
    cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

    // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
    //    In this example, the boundary conditions are defined by marking all
    //    the boundary attributes from the mesh as essential (Dirichlet) and
    //    converting them to a list of true dofs.
    Array<int> ess_tdof_list;
    if (mesh.bdr_attributes.Size())
    {
        Array<int> ess_bdr(mesh.bdr_attributes.Max());
        ess_bdr = 1;
        fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    // 7. Set up the linear form b(.) which corresponds to the right-hand side
    //    of the FEM linear system, which in this case is (f,phi_i) where f is
    //    given by the function f_exact and phi_i are the basis functions in the
    //    finite element fespace.
    Vector center(2);
    center[0] = -151.92;
    center[1] = 245.596;
    double epsilon = 2.8;
    double volume = M_PI * epsilon * epsilon;  // Area of the support region

    // TO DO: Make mesh_generator write in config file centre and epsilon
    // --- Source amplitude (in N/m² for example)
    double total_force = 1e6;  // Adjust this as needed

    VectorFunctionCoefficient source(dim, [&](const Vector& x, Vector& f)
        {
            f = 0.0;
            Vector diff = x;
            diff -= center;
            if (abs(diff.Norml2()) < epsilon)
            {
                // Dirac delta approximated as uniform over small region
                f[0] = total_force;
                f[1] = total_force;// Vertical force
            }
        });

    VectorFunctionCoefficient f(/*sdim, */source);
    LinearForm* b = new LinearForm(fespace);
    b->AddDomainIntegrator(new VectorDomainLFIntegrator(f));
    b->Assemble();
    saveSparseVectorInFile(b, configPath, "globalVectorMFEM");

    //std::cout << "Symmetry of M is " << IsSymmetric(M, tau/10);

    // 9. MINE
    // Set up mass and stiffness matrices
    BilinearForm m(fespace);
    real_t constRho = 2000.;
    Vector rho(&constRho, 1);
    PWConstCoefficient coeffRho(rho);
    m.AddDomainIntegrator(new VectorMassIntegrator(coeffRho));
    ConstantCoefficient alpha(0.45);
    //m.AddBoundaryIntegrator(new BoundaryMassIntegrator(alpha), 3);
    m.Assemble();
    m.Finalize();
    SparseMatrix M;
    m.FormSystemMatrix(ess_tdof_list, M);
    M.saveSparseMatrixInFile(configPath, "globalMassMFEM");

    ConstantCoefficient lambda(0.5 * 1e11);
    ConstantCoefficient mu(0.25 * 1e11);

    BilinearForm k(fespace);
    k.AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
    k.Assemble();
    k.Finalize();
    SparseMatrix K;
    k.FormSystemMatrix(ess_tdof_list, K);
    K.saveSparseMatrixInFile(configPath, "globalStiffMFEM");
    return 0;
    // 10. MINE- do i need to delete manually? set up block matrices lhs and rhs
    int size = m.NumRows();
    SparseMatrix I(size);
    for (int i = 0; i < size; i++)
    {
        I.Set(i, i, 1.0);
    }
    I.Finalize(); // Make it ready for use

    Array<int> block_offsets(3);  // For 2 blocks: 0, size1, size1+size2
    block_offsets[0] = 0;
    block_offsets[1] = size;
    block_offsets[2] = size + size;

    BlockMatrix rhs(block_offsets);
    BlockMatrix lhs(block_offsets);
    rhs.SetBlock(1, 1, new SparseMatrix(I));
    rhs.SetBlock(0, 0, new SparseMatrix(M));
    lhs.SetBlock(1, 1, new SparseMatrix(I));
    lhs.SetBlock(0, 0, new SparseMatrix(M));
    
    double tau = 0.002;
    I *= tau / 2;
    K *= tau / 2;

    rhs.SetBlock(1, 0, new SparseMatrix(I));
    rhs.SetBlock(0, 1, new SparseMatrix(K));

    lhs.SetBlock(1, 0, Add(0.0, I, -1.0, I));
    lhs.SetBlock(0, 1, Add(0.0, I, -1.0, K));

    std::cout << lhs.NumColBlocks() << " linear systems\n";

    // 10. Assemble the bilinear form and the corresponding linear system,
    //     applying any necessary transformations such as: eliminating boundary
    //     conditions, applying conforming constraints for non-conforming AMR,
    //     static condensation, etc.

    mfem::Array<int> block_offsetsV(3);
    // Fill block_offsets depending on how many blocks you have
    block_offsetsV[0] = 0;
    block_offsetsV[1] = size;
    block_offsetsV[2] = 2 * size;

    mfem::BlockVector globalVector(block_offsetsV);
    globalVector.GetBlock(0) = *b;
    Vector& nullV = globalVector.GetBlock(1);
    for (double& real : nullV)
    {
        real = 0;
    }

    BlockVector prevSol(block_offsets);    // sol^t
    BlockVector sol(block_offsetsV);  // sol^{t+1}
    sol.GetBlock(0) = nullV;
    sol.GetBlock(1) = nullV;
    prevSol = sol;

    // solve with direct solver
    mfem::DSmoother direct_solver;
    direct_solver.SetOperator(lhs.GetBlock(0,0));
    direct_solver.SetPositiveDiagonal(true);
    direct_solver.Mult(globalVector.GetBlock(0), sol.GetBlock(0)); // Solve lhs*sol=globalVector

    mfem::Vector temp(globalVector.GetBlock(0).Size());
    for (int iter = 0; iter < 1000; iter++) {
        direct_solver.Mult(globalVector.GetBlock(0), temp);
        sol.GetBlock(0) = temp;

        // Check residual periodically
        if (iter % 100 == 0) {
            mfem::Vector residual(globalVector.GetBlock(0).Size());
            lhs.GetBlock(0, 0).Mult(sol.GetBlock(0), residual);
            residual -= globalVector.GetBlock(0);
            double res_norm = residual.Norml2();
            double rel_res = res_norm / globalVector.GetBlock(0).Norml2();
            std::cout << "Iteration " << iter << ": Residual = " << res_norm
                << ", Relative residual = " << rel_res << std::endl;

            // Early stopping if convergence is good enough
            if (rel_res < 1e-10) break;
        }
    }
    //check residual

    Vector r = nullV;
    lhs.GetBlock(0,0).Mult(sol.GetBlock(0), r);
    r -= globalVector.GetBlock(0);
    std::cout << r.Norml2() << std::endl;
    std::cout << b->Norml2() << " " << sol.GetBlock(0).Norml2();

//    // 11. Solve the linear system A X = B.
//    std::cout << "\nUsing Generalized Minimal Residual Method!\n";
//    mfem::GMRESSolver solver;  // Create the GMRES solver object
//    mfem::HypreParMatrix* lhs_par = nullptr;
//    ConvertBlockMatrixToHypreParMatrix(lhs, lhs_par, mesh, *fespace, fec);
//    if (lhs_par->Height())
//    {
//        std::cout << ":)";
//    }
//    solver.SetOperator(lhs.GetBlock(0,0));   // Set the matrix (lhs) as the operator for the solver
//    solver.SetRelTol(1e-31);   // Set relative tolerance (adjust as needed)
//    solver.SetMaxIter(6300);    // Set maximum iterations (adjust as needed)
//    solver.SetPrintLevel(1);   // Set print level (1 for basic output, 2 for detailed)
//    solver.SetKDim(90); // Restart parameter
//
//    //TO DO: Start using Hypre preconditioners for faster performance
//    //TO DO: Make sure the method whichever it is converges
//    // TO DO: Maybe use a better preconditioner?
//    // Preconditioner (optional but strongly recommended)
//    //HypreBoomerAMG amg;
//    //amg.SetOperator(*lhs_par);
//    //solver.SetPreconditioner(amg);
//
//    mfem::DSmoother smoother;
//    smoother.SetOperator(lhs.GetBlock(0,0)); // your serial matrix
//    solver.SetPreconditioner(smoother);
//
//
//    std::ofstream outFile(configPath + "\\MFEM\\"+"all_solutionsMFEM.csv");
//
//    std::cout << "||b|| = " << globalVector.GetBlock(0).Norml2() << std::endl;
//    std::cout << "||x|| = " << sol.GetBlock(0).Norml2() << std::endl;
//    //lhs.GetBlock(0, 0).Print(std::cout, 1); // prints top-left block
//
//    solver.Mult(globalVector.GetBlock(0), sol.GetBlock(0));
//    
//    
//    //Vector r = nullV;
//    //lhs.GetBlock(0,0).Mult(sol.GetBlock(0), r);
//    //r -= globalVector.GetBlock(0);
//    //std::cout << r.Norml2() << std::endl;
//
//
//    // Assume lhs, rhs are already BlockOperators or Operators
//// Assume y is a Vector (or BlockVector) of compatible size
//
//    mfem::BlockVector rhs_vec(block_offsets);
//    double Tmax = 0.2;
//    int stepsCount = ceil(Tmax / tau);
//    for (int i = 0; i < stepsCount; i++)
//    {
//        std::cout << "it " << i << "\n";
//        // rhs_vec = rhs*prevSol
//        rhs.Mult(prevSol, rhs_vec);
//        // Solve the system
//        solver.Mult(rhs_vec, sol);
//        printCurrentSol(sol, outFile);
//    }
//




//
//    // 12. Recover the solution as a finite element grid function.
//    a->RecoverFEMSolution(X, *b, x);
//
//    // 13. Compute and print the L^2 norm of the error.
//    cout << "\n|| E_h - E ||_{L^2} = " << x.ComputeL2Error(E) << '\n' << endl;
//
//    // 14. Save the refined mesh and the solution. This output can be viewed
//    //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
//    {
//        ofstream mesh_ofs("refined.mesh");
//        mesh_ofs.precision(8);
//        mesh.Print(mesh_ofs);
//        ofstream sol_ofs("sol.gf");
//        sol_ofs.precision(8);
//        x.Save(sol_ofs);
//    }
//
//    // 15. Send the solution by socket to a GLVis server.
//    if (visualization)
//    {
//        char vishost[] = "localhost";
//        int  visport = 19916;
//        socketstream sol_sock(vishost, visport);
//        sol_sock.precision(8);
//        sol_sock << "solution\n" << mesh << x << flush;
//    }
//
//    // 16. Free the used memory.
//    delete a;
//    delete sigma;
//    delete muinv;
//    delete b;
//    delete fespace;
//    delete fec;

    return 0;
}

void printCurrentSol(const mfem::BlockVector& sol, std::ofstream& outFile)
{
    for (int b = 0; b < sol.NumBlocks(); ++b)
    {
        const mfem::Vector& block = sol.GetBlock(b);
        for (int i = 0; i < block.Size(); ++i)
        {
            outFile << block[i];
            if (b < sol.NumBlocks() - 1 || i < block.Size() - 1)
            {
                outFile << ",";
            }
        }
    }
    outFile << "\n";  // newline after each time step, if relevant
}

//void PrintSparseMatrixToFile(const mfem::SparseMatrix& mat, const std::string& filename)
//{
//    std::ofstream out(filename);
//    if (!out)
//    {
//        std::cerr << "Error opening file: " << filename << std::endl;
//        return;
//    }
//
//    out << "# Format: row, col, value\n";
//    for (int i = 0; i < mat.Height(); ++i)
//    {
//        int start = mat.GetRowStarts()[i];
//        int end = mat.GetRowStarts()[i + 1];
//        for (int j = start; j < end; ++j)
//        {
//            int col = mat.GetColIndices()[j];
//            double val = mat.GetData()[j];
//            out << i << "," << col << "," << val << "\n";
//        }
//    }
//
//    out.close();
//}

//void saveSparseMatrixInFile(const std::string& filename, const SparseMatrix& M)
//{
//    std::ofstream matrixStream(filename);
//    auto massI = M.GetI();
//    auto massJ = M.GetJ();
//    auto massA = M.GetData();
//    int prevI = 0;
//    for (int i = 1; i < M.Height() + 1; i++)
//    {
//        int I = massI[i];
//        for (int j = prevI; j < I; j++)
//        {
//            matrixStream << prevI << ", " << massJ[j] << ", " << massA[j] << "\n";
//        }
//        matrixStream << std::endl;
//        prevI = I;
//    }
//}



void AssembleSum(const mfem::BlockMatrix& A, const mfem::BlockMatrix& B, mfem::BlockMatrix& Result)
{
    MFEM_VERIFY(A.NumRowBlocks() == B.NumRowBlocks(), "Row block count mismatch!");
    MFEM_VERIFY(A.NumColBlocks() == B.NumColBlocks(), "Column block count mismatch!");

    for (int i = 0; i < A.NumRowBlocks(); i++)
    {
        for (int j = 0; j < A.NumColBlocks(); j++)
        {
            // Sum the two blocks
            mfem::SparseMatrix* sum = Add(A.GetBlock(i, j), B.GetBlock(i, j));
            Result.SetBlock(i, j, sum);
        }
    }
}

void AssembleDiff(const mfem::BlockMatrix& A, const mfem::BlockMatrix& B, mfem::BlockMatrix& Result)
{
    MFEM_VERIFY(A.NumRowBlocks() == B.NumRowBlocks(), "Row block count mismatch!");
    MFEM_VERIFY(A.NumColBlocks() == B.NumColBlocks(), "Column block count mismatch!");

    for (int i = 0; i < A.NumRowBlocks(); i++)
    {
        for (int j = 0; j < A.NumColBlocks(); j++)
        {
            // Sum the two blocks
            mfem::SparseMatrix* sum = Add(1.0, A.GetBlock(i, j), -1.0, B.GetBlock(i, j));// A + (-1) * B
            Result.SetBlock(i, j, sum);
        }
    }
}

void E_exact(const Vector& x, Vector& E)
{
    if (dim == 3)
    {
        E(0) = sin(kappa * x(1));
        E(1) = sin(kappa * x(2));
        E(2) = sin(kappa * x(0));
    }
    else
    {
        E(0) = sin(kappa * x(1));
        E(1) = sin(kappa * x(0));
        if (x.Size() == 3) { E(2) = 0.0; }
    }
}

void f_exact(const Vector& x, Vector& f)
{
    if (dim == 3)
    {
        f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
        f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
        f(2) = (1. + kappa * kappa) * sin(kappa * x(0));
    }
    else
    {
        f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
        f(1) = (1. + kappa * kappa) * sin(kappa * x(0));
        if (x.Size() == 3) { f(2) = 0.0; }
    }
}
