#include "ElasticFEMAssembler.h"
#include "FilesHandler.h"
#include "ConfigParser.h"

#pragma once

constexpr double f0 = 5;
constexpr int sourcePower = 1e2;

void printVectorS(const std::string& fileName, const Eigen::SparseVector<double>& vecForPrint, bool turnedOn)
{
    if (!turnedOn)
        return;

    std::ofstream outFile(fileName);
    for (int i = 0; i < vecForPrint.size(); ++i)
    {
        outFile << vecForPrint.coeff(i);
        if (i < vecForPrint.size() - 1)
            outFile << ",";
    }
    outFile << "\n";
}

void printVectorS(std::ofstream& outFile, const Eigen::SparseVector<double>& vecForPrint, bool turnedOn)
{
    if (!turnedOn)
        return;

    for (int i = 0; i < vecForPrint.size(); ++i)
    {
        outFile << vecForPrint.coeff(i);
        if (i < vecForPrint.size() - 1)
            outFile << ",";
    }
    outFile << "\n";
}

void printVector(std::ofstream& outFile, const DenseVector& vecForPrint, bool turnedOn)
{
    if (!turnedOn)
        return;

    for (int i = 0; i < vecForPrint.size(); ++i)
    {
        outFile << vecForPrint.coeff(i);
        if (i < vecForPrint.size() - 1)
            outFile << ",";
    }
    outFile << "\n";
}

void printProgress(bool printTimeStepProgress, int printStepsThreshhold, int t)
{
    if (printTimeStepProgress)
    {
        std::cout << "Finished time step " << t << '\n';
    }
    else if (t % printStepsThreshhold == 0)
    {
        std::cout << "On time step " << t << "\n";
    }
}

double EstimateConditionNumberSparse(const Eigen::SparseMatrix<double>& A)
{
    // For sparse matrices, we can use iterative eigensolvers

    // Estimate largest eigenvalue using power iteration
    Eigen::SparseMatrix<double> AtA = A.transpose() * A;
    int n = A.cols();
    Eigen::VectorXd x = Eigen::VectorXd::Random(n);

    // Power iteration for largest eigenvalue
    for (int i = 0; i < 100; i++) {
        x = AtA * x;
        x.normalize();
    }
    double lambda_max = (x.transpose() * (AtA * x))(0);

    // For smallest eigenvalue, we can use inverse iteration
    // or use a library like Spectra for more robust sparse eigenvalue computation

    // Simple estimation using the Frobenius norm
    double frobenius_norm = std::sqrt((AtA).coeffs().abs().sum());
    double condition_estimate = frobenius_norm / std::numeric_limits<double>::epsilon();

    return condition_estimate;
}

// Poor man's PML- set the displacements at the artificial border to 0.
void dampBorder(DenseVector& sol, const std::vector<int>& fakeNds)
{
    for (int nodeInd : fakeNds)
    {
        sol.coeffRef(nodeInd) = 0.0;
    }
}

// Initial condition
double ricker(double t)
{
    double t_shift = 2 / f0;
    double a = M_PI * f0 * (t - t_shift);
    return (1 - 2 * a * a) * exp(-a * a);
}

double euclideanDistanceSquared(const double& x0, const double& y0, const double& x, const double& y)
{
    return pow(x0 - x, 2) + pow(y0 - y, 2);
}

double euclideanDistance(const double& x0, const double& y0, const double& x, const double& y)
{
    return sqrt(euclideanDistanceSquared(x0, y0, x, y));
}

std::vector<int> resolveSourceIndices(const std::vector<std::pair<double, double>>& nodes, const std::vector<double>& sourceEpicentre, const double& sourceRadius)
{
    std::vector<int> res;
    for (int i = 0; i < nodes.size(); i++)
    {
        /*sqrt(pow(sourceEpicentre[0] - nodes[i].first, 2) + pow(sourceEpicentre[1] - nodes[i].second, 2))*/
        if (euclideanDistance(sourceEpicentre[0], sourceEpicentre[1], nodes[i].first, nodes[i].second) <= sourceRadius)
        {
            res.emplace_back(i);
            std::cout << "Adding node " << i << " as a source node!\n";
        }
    }
    return res;
}

double calculateSourceVx(const double& x, const double& y, const std::vector<double>& sourceEpicentre, const double& sourceRadius)
{
    double distY = y - sourceEpicentre[1];
    double dist = euclideanDistanceSquared(sourceEpicentre[0], sourceEpicentre[1], x, y);
    return -2 * distY * exp(-dist/pow(sourceRadius,2));
}

double calculateSourceVy(const double& x, const double& y, const std::vector<double>& sourceEpicentre, const double& sourceRadius)
{
    return -2 * (x - sourceEpicentre[0]) * exp(-euclideanDistanceSquared(sourceEpicentre[0], sourceEpicentre[1], x, y) / pow(sourceRadius, 2));
}

// Eigen to MFEM change means xyxy to xxyy change of indices
int getMFEMSPosition(int row, int halfSize)
{
    return row / 2 + (row % 2 == 1) * halfSize;
}

// MFEM to Eigen change means xxyy to xyxy change of indices
int getEigenPosition(int row, int halfSize)
{
    if (row >= halfSize)
    {
        return (row - halfSize) * 2 + 1;
    }
    return row * 2;
}

int identity(int row, int halfSize)
{
    return row;
}

// Reorder the rows and cols from Eigen format to MFEM and vice versa
void convertEigenFromToMFEM(TripletsList& matrix, int halfSize, const std::function<int(int, int)>& comparator, bool updateIndices = true)
{
    if (!updateIndices)
    {
        std::sort(matrix.begin(), matrix.end(), [&halfSize, &comparator](const T& right, const T& left)
            {
                int rightActualRow = comparator(right.row(), halfSize);
                int leftActualRow = comparator(left.row(), halfSize);
                if (rightActualRow == leftActualRow)
                    return comparator(right.col(), halfSize) < comparator(left.col(), halfSize);
                return rightActualRow < leftActualRow;
            }
        );
    }
    else
    {
        TripletsList temp;
        temp.reserve(matrix.size());
        for (const T& triplet : matrix)
        {
            temp.emplace_back(comparator(triplet.row(), halfSize), comparator(triplet.col(), halfSize), triplet.value());
        }
        matrix = temp;
        std::sort(matrix.begin(), matrix.end(), [](const T& right, const T& left)
            {
                if (right.row() == left.row())
                    return right.col() < left.col();
                return right.row() < left.row();
            }
        );
    }

}

void lumpMassMatrix(TripletsList& M)
{
    for (auto& triplet : M)
    {
        triplet = Eigen::Triplet<double>(triplet.row(), triplet.row(), triplet.value());
    }
}

// TO DO: Check whether zeroes from the local matrices (localM0, prrotoM1) have been converted to non zero elements of the global matrix.
void removeFactualZeroes(const std::vector<std::vector<int>>& elements, TripletsList& M, TripletsList& K)
{
    std::map<std::pair<int, int>, bool> edgesOccuraances;
    for (auto element : elements)
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                edgesOccuraances[{element[i], element[j]}] = true;
                edgesOccuraances[{element[j], element[i]}] = true;
            }
        }
    }
    for (auto& triplet : M)
    {
        std::pair<int, int> a = { floor(triplet.row() / 2), floor(triplet.col()) / 2 };
        if (edgesOccuraances[a] == 0)
        {
            triplet = Eigen::Triplet<double>();
            std::cout << "Element of M at row " << triplet.row() << " and col " << triplet.col() << " should be zero but isn't!\n";
        }
    }
    for (auto& triplet : K)
    {
        std::pair<int, int> a = { floor(triplet.row() / 2), floor(triplet.col()) / 2 };
        if (edgesOccuraances[a] == 0)
        {
            triplet = Eigen::Triplet<double>();
            std::cout << "Element of K at row " << triplet.row() << " and col " << triplet.col() << " should be zero but isn't\n";
        }
    }
}

std::vector<int> getFakeBoundaries(const std::vector<std::pair<double, double>>& nodes, const std::vector<double>& depths,
                                   const double& left, const double& right, const std::string& modelProblem)
{
    std::set<int> res;
    double tol = 0.01;
    for (int i = 0; i < nodes.size(); i++)
    {
        // This way we don't have to worry about cavities being damped
        if (abs(nodes[i].second - depths[0]) < tol || abs(nodes[i].first - left) < tol || abs(nodes[i].first - right) < tol)
        {
            res.emplace(i);
        }
        else if (modelProblem == "Analytical" && abs(nodes[i].second - depths[2]) < tol)
        {
            res.emplace(i);
        }
    }
    std::vector<int> vecRes;
    for (int node : res)
    {
        vecRes.push_back(2 * nodes.size() + 2 * node);
        vecRes.push_back(2 * nodes.size() + 2 * node + 1);
    }
    return vecRes;
}
