#define _USE_MATH_DEFINES 
#include <chrono>
#include "utils.h"


void assembleGlobalMatricesSeismic2D(const TripletsList& M0, const TripletsList& M1, TripletsList vec, int dim, 
                                     SparseMatrix& lhs, SparseMatrix& rhs, Eigen::SparseVector<double>& b, double tau, 
                                     const std::string& modelProblem)
{
    /// rho*(M0,0)(p_i+1-pi)=tau*(F)+tau/2*(0 -M1)(p_i+1+pi)
    ///     (0, E)(q_i-1qi)      (0)       (E   0)(q_i+1+qi)
    ///                         |                                     
    ///                         V
    /// (rho*M0+tau/2*M1 0)(p_i+1)=tau*(F)+(rho*M0 -tau/2*M1 0)(pi)
    /// (-tau/2*E        E)(q_i+1)     (0) (tau/2*E          E)(qi)
    TripletsList matrix2;
    TripletsList matrix1;
    matrix1.reserve(M0.size() + dim);
    matrix2.reserve(M0.size() + dim);
    std::transform(M0.begin(), M0.end(), std::back_inserter(matrix1),
        [](const T& triplet) {
            return T(triplet.row(), triplet.col(), triplet.value());
        });

    std::transform(M1.begin(), M1.end(), std::back_inserter(matrix2),
        [&dim, &tau](const T& triplet) {
            return T(triplet.row(), triplet.col() + dim, -0.5 * tau * triplet.value());
        });

    for (int i = 0; i < dim; i++)
    {
        matrix1.emplace_back(dim + i, dim + i, 1);
        matrix2.emplace_back(dim + i, i, 0.5 * tau);
    }

    SparseMatrix A(2 * dim, 2 * dim);
    A.setFromTriplets(matrix1.begin(), matrix1.end());
    SparseMatrix B(2 * dim, 2 * dim);
    B.setFromTriplets(matrix2.begin(), matrix2.end());
    lhs = A - B;
    rhs = A + B;
    std::cout << "Equation matrices have been set up!\n";


    // for analytical solutions we have displacement initial conditions
    SparseMatrix V(dim * 2, 1);
    if (modelProblem == "Analytical")
    {
        for (int i = 0; i < vec.size(); i++)
        {
            vec[i] = T(vec[i].row() + dim, vec[i].col(), vec[i].value());
        }
        V.setFromTriplets(vec.begin(), vec.end());
    }
    else
    {
        // M0.f for the source
        SparseMatrix M(2 * dim, 2 * dim);
        M.setFromTriplets(M0.begin(), M0.end());

        V.setFromTriplets(vec.begin(), vec.end());
        V = M * V;
    }
    b = V;

    printVectorS("D:\\Serious\\FMI\\Магистър\\Дипломен семестър\\FEM2D_config\\initialCond", b, false);
}

void euler(SparseMatrix& lhs, const SparseMatrix& rhs, Eigen::SparseVector<double>& b, int dim, const ConfigParser& cfgParser,
           const ElasticFEMAssembler& assembler, const std::string& configPath, const std::string& modelProblem)
{
    std::cout << "Starting euler method...\n";
    std::cout << "Factorizing matrix with dimensions " << lhs.rows() << ", " << lhs.cols() << "and " << lhs.nonZeros() << " non zero elements" << '\n';
    int stepsCount = ceil(cfgParser.Tmax / cfgParser.tau);
    DenseVector sol(2 * dim);
    sol.setZero();

    // monitor integral of ricker wavelet
    double total = 0;
    DenseVector prevSol(2*dim);
    assert(lhs.rows() == b.size());
    Eigen::SparseLU<SparseMatrix> solver;
    lhs.makeCompressed();
    solver.analyzePattern(lhs);
    solver.factorize(lhs);
    if (solver.info() != Eigen::Success) {
        std::cout << solver.lastErrorMessage();
        std::cerr << "Factorization failed!" << std::endl;
        return;
    }
    std::cout << "Factorization was successful!";
    int printStepsThreshhold = 100;
    bool printTimeStepProgress = stepsCount < printStepsThreshhold;
    std::cout << "Print time stepping is set to "  << printTimeStepProgress << '\n';

    std::string displacementFile = cfgParser.createOutputFilename(configPath, false);
    std::ofstream outFile(displacementFile);
    std::ofstream rckr(configPath + "\\ricker");
    std::ofstream snapshots(cfgParser.createOutputFilename(configPath, true));
    int stepsPerSnapshot = 1;
    if (cfgParser.allNodes > 0)
    {
        stepsPerSnapshot = stepsCount / cfgParser.allNodes;
    }

    std::cout << "Will compute " << stepsCount << " steps and will make " << stepsPerSnapshot << " steps per snapshot\n";
    auto start = std::chrono::high_resolution_clock::now();
    const  Eigen::SparseVector<double> bConst = b;
    Eigen::SparseVector<double> bPrev = sourcePower * bConst * ricker(0);
    for (int t = 0; t <= stepsCount + 2; t++)
    {
        if (cfgParser.allNodes > 0)
        {
            //This prints all nodes on each time step- leave for debugging
            printVector(outFile, sol, false);
            // Debugging purposes for Analytical mode- check if initial condition is set up correctly
            printVectorS(outFile, b, false);

            // I want to have the displacement at the first moment as well
            if (t % stepsPerSnapshot == 1 )
            {
                printVector(snapshots, sol, true);
                std::cout << "Saving at step " << t << "\n";
                // Probably was used for debugging
                printVectorS(snapshots, b, false);
            }
        }
        else
        {
            for (int i = 0; i < cfgParser.seismographIndices.size(); ++i)
            {
                outFile << sol[cfgParser.seismographIndices[i]];
                if (i < cfgParser.seismographIndices.size() - 1)
                    outFile << ",";
            }
            outFile << "\n";
        }
        prevSol = sol;

        //for model problem that is not Analytical we need b not only at t=0
        if (modelProblem != "Analytical")
        {
            const std::vector<int> sourceRotatorCurl = { 1,0,0,1,-1,-1 };
            const std::vector<int> sourceRotator= { 1,1,1,1,1,1 };
            rckr << sourcePower << ", ";
            //auto bPtr = b.valuePtr();
            double a = sourcePower * ricker(cfgParser.tau * (t + 1));
            total += cfgParser.tau * a;
            /*for (int i = 0; i < b.data().size(); i++)
            {
                *(bPtr + i) = a;// *sourceRotatorCurl[i];
            }*/
            b = a * bConst;
            // The code below is used in case of non constant source function
            //for (int k = 0; k < b.nonZeros(); k += 2)
            //{
            //    double value = bPtr[k];
            //    int index = b.innerIndexPtr()[k] / 2;
            //    // set vx and vy to curl components of a function- that way we will have zero divergence
            //    *(bPtr + k) = a;// *calculateSourceVx(assembler.nodes[index].first, assembler.nodes[index].second, cfgParser.sourceEpicentre, cfgParser.sourceRadius);
            //    *(bPtr + k + 1) = a;// *calculateSourceVy(assembler.nodes[index].first, assembler.nodes[index].second, cfgParser.sourceEpicentre, cfgParser.sourceRadius);
            //}
        }
        if (t == 0)
        {
            if (modelProblem == "Analytical")
            {
                sol = b;
                b.setZero();
            }
            else
            {
                // initial displ and vel are both zero
                sol = solver.solve((b + bPrev) * cfgParser.tau / 2);// NOTE: THIS WILL MESS UP ANALYTICAL SOLUTIONS :(
                auto r = lhs * sol - (b + bPrev) * cfgParser.tau / 2;// REVERT THESE 2 LINES TO USE ONLY b
                double residual_norm = r.norm();
                double relative_residual = residual_norm / b.norm();
                std::cout << "b norm is " << b.norm() << ", b nonzeroes are " << b.nonZeros() << ", residual norm is " << residual_norm << " and relative residual norm is " << relative_residual << '\n';
            }
        }
        else
        {
            auto bb = rhs * prevSol + (b + bPrev) * cfgParser.tau / 2;
            sol = solver.solve(bb);
            // Зануляваме граничните възли на sol
            dampBorder(sol, cfgParser.fakeBoundary);

            /*auto r = lhs * sol - bb;
            double residual_norm = r.norm();
            double relative_residual = residual_norm / bb.norm();
            std::cout << "residual norm is " << residual_norm << " and relative residual norm is " << relative_residual;*/
        }
        bPrev = b;
        printProgress(printTimeStepProgress, printStepsThreshhold, t);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
    std::cout << "Total energy from the ricker wavelet is " << total;
}

int main() {
    std::string configPath = "D:\\Serious\\FMI\\Магистър\\Дипломен семестър\\FEM2D_config";
    FilesHandler filesHandler(configPath);
    int dim = filesHandler.nodes.size() * 2;// for 2D case
    ConfigParser cfgParser(filesHandler.readConfig(), dim);
    std::cout << "Read config file\n";
    TripletsList mat1;
    TripletsList mat0;
    TripletsList vec;
    
    // Two ways to implement the source- first one is by indices of source elements, second one is hipocentre and radius of the source
    // First one is with assembleGlobalVector second one is with assembleGlobalVectorFromNodes and is not correct FEM-wise
    std::vector<int> sourceIndices;
    if (cfgParser.modelProblem == "Analytical")
    {
        sourceIndices = resolveSourceIndices(filesHandler.nodes, cfgParser.sourceEpicentre, cfgParser.sourceRadius);
    }
    else
    {
        sourceIndices = cfgParser.sourceIndices;
    }
    std::vector<int> seismographIndices = cfgParser.seismographIndices;
    Eigen::SparseVector<double> b;
    ElasticFEMAssembler assembler(std::move(filesHandler.nodes), std::move(filesHandler.elements), cfgParser.depths, cfgParser.left, 
                                  cfgParser.right, cfgParser.cavityFilling, cfgParser.modelProblem, cfgParser.cavityIndices);

    assembler.assembleGlobalVector(sourceIndices);
    cfgParser.fakeBoundary = getFakeBoundaries(assembler.nodes, cfgParser.depths, cfgParser.left, cfgParser.right, cfgParser.modelProblem);
    if (cfgParser.debug)
    {
        std::cout << "Assembling...\n";
        assembler.assembleGlobalMassMatrix();
        assembler.assembleGlobalStiffnessMatrix();
        mat1 = assembler.K;
        mat0 = assembler.M;
        convertEigenFromToMFEM(mat0, dim / 2, identity);
        convertEigenFromToMFEM(mat1, dim / 2, identity);
        filesHandler.saveSparseMatrixInFileMFEMStyle("globalMassEigen", mat0);
        filesHandler.saveSparseMatrixInFileMFEMStyle("globalStiffEigen", mat1);
        if (cfgParser.useMFEMMatrices)
        {
            std::cout << "Reading MFEM Matrices...\n";
            filesHandler.readMFEMGlobalMatrices(mat0, mat1, vec);
            convertEigenFromToMFEM(mat0, dim / 2, getEigenPosition);
            convertEigenFromToMFEM(mat1, dim / 2, getEigenPosition);

            convertEigenFromToMFEM(mat0, dim / 2, getMFEMSPosition);
            filesHandler.saveSparseMatrixInFileMFEMStyle("globalMassEigenTestConverter", mat0);
            convertEigenFromToMFEM(mat1, dim / 2, getMFEMSPosition);
            filesHandler.saveSparseMatrixInFileMFEMStyle("globalStiffEigenTestConverter", mat1);
        }
        return 0;
    }
    else if(cfgParser.useMFEMMatrices)
    {
        std::cout << "Reading MFEM matrices...";
        filesHandler.readMFEMGlobalMatrices(mat0, mat1, vec);
        convertEigenFromToMFEM(mat0, dim / 2, getEigenPosition);
        convertEigenFromToMFEM(mat1, dim / 2, getEigenPosition);
    }
    else
    {
        std::cout << "Assembling...\n";
        assembler.assembleGlobalMassMatrix();
        assembler.assembleGlobalStiffnessMatrix();
        mat1 = assembler.K;
        //assembler.lumpMassMatrix();
        mat0 = assembler.M;

        //assembler.removeFactualZeroes();
    }

    vec = assembler.V;
    SparseMatrix lhs;
    SparseMatrix rhs;
    assembleGlobalMatricesSeismic2D(mat0, mat1, vec, dim, lhs, rhs, b, cfgParser.tau, cfgParser.modelProblem);
    euler(lhs, rhs, b, dim, cfgParser, assembler, configPath, cfgParser.modelProblem);

    return 0;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// WARNING: THIS INVERTS A SPARSE MATRIX, CAN BE UNSTABLE AND DOES NOT BENEFIT FROM IT'S STRUCTURE!
// USE assembleGlobalMatricesSeismic2D INSTEAD
//void generateFirstOrderPDE(const SparseMatrix& M0, const SparseMatrix& M1, const Eigen::SparseVector<double>& b,
//    DenseMatrix& M, DenseVector& v)
//{
//    Eigen::SparseLU<SparseMatrix> solver;
//    solver.analyzePattern(M0);
//    solver.factorize(M0);
//
//    DenseVector x = solver.solve(b);
//    DenseMatrix X = solver.solve(M1);
//    int dim = X.rows();
//    //DenseMatrix M(2 * dim, 2 * dim);
//    M.setZero();
//    M.block(0, dim, dim, dim) = X;
//    M.block(dim, 0, dim, dim) = DenseMatrix::Identity(dim, dim);
//
//    //Eigen::VectorXd v(2 * dim);
//    v.tail(dim).setZero();
//    v.head(dim) = x;
//}
