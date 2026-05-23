#include <iostream>
#include <cmath>
#include <set>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include<Eigen/SparseCholesky>
#include<Eigen/SparseLU> 

#pragma once

using T = Eigen::Triplet<double>;
using TripletsList = std::vector<T>;
using DenseMatrix = Eigen::MatrixXd;
using DenseVector = Eigen::VectorXd;
using SparseMatrix = Eigen::SparseMatrix<double>;

class Media {
    static double lambda;
    static double mu;
};

class Moho : public Media {
public:
    // NOTE: These parameters are modified with the purpose of using km distances as opposed to m

    static constexpr double rhoMantle = 3e12;
    static constexpr double lambda = 0.76e14;
    static constexpr double mu = 0.66e14;
    static double getRho() {
        return rhoMantle;
    }
};

class Rock : public Media {
public:
    // NOTE: These parameters are modified with the purpose of using km distances as opposed to m
    //static constexpr double rhoRock = 2e12;

    //static constexpr double lambda = 0.5e14;
    //static constexpr double mu = 0.25e14;

    // NOTE: These parameters are modified with the purpose of using m distances as opposed to km
    static constexpr double rhoRock = 2000;//2650;// analytical s wave= 3e12;//2000.

    static constexpr double lambda = 0.5e11;//225 * rhoRock * 1e4;// analytical s wave- 0.25e14;//0.5e11
    static constexpr double mu = 0.25e11;/*9 * rhoRock * 1e6;*/// analytical s wave0.25e14;//0.25e11
    static double getRho(){
        return rhoRock;
    }
};

class LinearIce : public Media {
public:
    static constexpr double lambda = 27 / 4 * 1e9;
    static constexpr double mu = 27 / 8 * 1e9;
    static double getRho(const std::vector<std::pair<double, double>>& nodes, const std::vector<int>& element, const std::vector<double>& depths){
        double medicentreHeight = (nodes[element[0]].second + nodes[element[1]].second + nodes[element[2]].second) / 3;
        if (medicentreHeight > depths[1])
        {
            return rhoLowerIce - (medicentreHeight - depths[1]) * (rhoLowerIce - rhoUpperIce) / (depths[2] - depths[1]);//600
        }
    }

    static constexpr double rhoLowerIce = 900.;
    static constexpr double rhoUpperIce = 300.;
};

class SquaredIce : public Media {
public:
    static constexpr double lambda = 27 / 4 * 1e9;
    static constexpr double mu = 27 / 8 * 1e9;
    static double getRho(const std::vector<std::pair<double, double>>& nodes, const std::vector<int>& element, const std::vector<double>& depths) {
        double medicentreHeight = (nodes[element[0]].second + nodes[element[1]].second + nodes[element[2]].second) / 3;
        if (medicentreHeight > depths[1])
        {
            return rhoUpperIce + sqrt((medicentreHeight - depths[2])*3000);
        }
    }

    static constexpr double rhoLowerIce = 900.;
    static constexpr double rhoUpperIce = 300.;
};

class Water : public Media {
public:
    static constexpr double lambda = 2 * 1e9;
    static constexpr double mu = 0;
    static double getRho(){
        return rhoWater;
    }
    static constexpr double rhoWater = 997.;
};

class Analytical : public Media {
public:
    static constexpr double lambda = 1.75e12;
    static constexpr double mu = 0.25e12;
    static double getRho() {
        return rhoAnalytical;
    }
    static constexpr double rhoAnalytical = 1e12;
    //static constexpr double rhoAnalytical = 1e12;
};

class ElasticFEMAssembler
{
public:
    DenseMatrix localM0;
    DenseMatrix protoM1;
    DenseVector localVector;

    int degreesOfFreedom;
    std::vector<std::pair<double, double>> nodes;
    std::vector<std::vector<int>> elements;
    std::vector<double> depths;
    int variablesCount;
    double left;
    double right;

    std::string cavityFilling;
    std::string modelProblem;
    std::vector<int> cavityIndices;

    TripletsList M;
    TripletsList K;
    TripletsList V;

    ElasticFEMAssembler(std::vector<std::pair<double, double>>&& nds, std::vector<std::vector<int>>&& elems, const std::vector<double>& _depths,
                        double _left, double _right, std::string const& _cavityFilling, const std::string& _modelProblem, const std::vector<int>& _cavityIndices) :
        nodes(std::move(nds)), elements(std::move(elems)), depths(_depths), left(_left), right(_right), cavityFilling(_cavityFilling),
        modelProblem(_modelProblem), cavityIndices(_cavityIndices)
    {
        initialize2DLinearElements();
    }

    void initialize2DLinearElements()
    {
        variablesCount = 2;
        DenseMatrix localm0(6, 6);
        localm0 << 1.0 / 12, 0., 1.0 / 24, 0., 1.0 / 24, 0.,
            0., 1.0 / 12, 0., 1.0 / 24, 0., 1.0 / 24,
            1.0 / 24, 0., 1.0 / 12, 0., 1.0 / 24, 0.,
            0., 1.0 / 24, 0., 1.0 / 12, 0., 1.0 / 24,
            1.0 / 24, 0., 1.0 / 24, 0., 1.0 / 12, 0.,
            0., 1.0 / 24, 0., 1.0 / 24, 0., 1.0 / 12;
        localM0 = localm0;
        DenseMatrix protom1(4, 6);
        protom1 << -1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            -1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, -1.0, 0.0, 1.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0, 0.0, 1.0;
        protoM1 = protom1;
        degreesOfFreedom = protoM1.cols() / 2;
        if (modelProblem != "Analytical")
        {
            DenseVector localvector(6);
            localvector << 1e10, 1e10, 1e10, 1e10, 1e10, 1e10;
            //localvector << 1e7, 1e7, 1e7, 1e7, 1e7, 1e7;
            localVector = localvector;
        }
        else
        {
            localVector = DenseVector(nodes.size() * 2);
            for (int i = 0; i < nodes.size(); ++i)
            {
                std::pair<double, double> initialAtNode = initialConditionsAnalyticalModel(nodes[i].first, nodes[i].second);
                localVector[variablesCount * i] = initialAtNode.first;
                localVector[variablesCount * i + 1] = initialAtNode.second;
            }
            std::cout << "";
        }

    }

    // TO DO: Implement case for acoustic wave equation- maybe it should not be here but in the upper functions?
    double getRho(const std::vector<int>& element, int elementNum) const
    {
        if (modelProblem == "Analytical")
            return Analytical::getRho();
        if(modelProblem == "Halfspace")
            return Rock::getRho();
        if (modelProblem == "Elastic" && cavityFilling == "w")
        {
            if (std::find(cavityIndices.begin(), cavityIndices.end(), elementNum) != cavityIndices.end())
            {
                return Water::getRho();
            }
        }
        double medicentreHeight = (nodes[element[0]].second + nodes[element[1]].second + nodes[element[2]].second) / 3;
        if (medicentreHeight > depths[1])
        {
            return LinearIce::getRho(nodes, element, depths);
        }
        return Rock::getRho();
    }

    double getMu(const std::vector<int>& element, int elNum) const
    {
        if (modelProblem == "Analytical")
            return Analytical::mu;
        if (modelProblem == "Halfspace")
            return Rock::mu;
        if (modelProblem == "Elastic" && cavityFilling == "w")
        {
            if (std::find(cavityIndices.begin(), cavityIndices.end(), elNum) != cavityIndices.end())
            {
                return Water::mu;
            }
        }
        double medicentreHeight = (nodes[element[0]].second + nodes[element[1]].second + nodes[element[2]].second) / 3;
        if (medicentreHeight > depths[1])
        {
            return LinearIce::mu;
        }
        return Rock::mu;
    }

    double getLambda(const std::vector<int>& element, int elNum) const
    {
        if (modelProblem == "Analytical")
            return Analytical::lambda;
        if (modelProblem == "Halfspace")
            return Rock::lambda;
        if (modelProblem == "Elastic" && cavityFilling == "w")
        {
            if (std::find(cavityIndices.begin(), cavityIndices.end(), elNum) != cavityIndices.end())
            {
                return Water::lambda;
            }
        }
        double medicentreHeight = (nodes[element[0]].second + nodes[element[1]].second + nodes[element[2]].second) / 3;
        if (medicentreHeight > depths[1])
        {
            return LinearIce::lambda;
        }
        return Rock::lambda;
    }

    DenseMatrix calculateLocalStiffnessMatrix(const std::vector<int>& element, int elNum)
    {
        std::vector<std::pair<double, double>> dots;
        dots.reserve(degreesOfFreedom);
        for (int df = 0; df < degreesOfFreedom; df++)
        {
            dots.emplace_back(nodes[element[df]]);
        }
        double lambda= getLambda(element, elNum);
        double mu= getMu(element, elNum);
        DenseMatrix j(2, 2);
        j << dots[1].first - dots[0].first, dots[1].second - dots[0].second,
            dots[2].first - dots[0].first, dots[2].second - dots[0].second;
        DenseMatrix jInv = j.inverse();
        DenseMatrix jacobianPart(3, 4);
        jacobianPart << jInv(0, 0), jInv(0, 1), 0, 0,
            0, 0, jInv(1, 0), jInv(1, 1),
            jInv(1, 0), jInv(1, 1), jInv(0, 0), jInv(0, 1);
        DenseMatrix d(3, 3);
        d << 2 * mu+ lambda, lambda, 0,
            lambda, 2 * mu+ lambda, 0,
            0, 0, mu;
        DenseMatrix B = jacobianPart * protoM1;
        return protoM1.transpose() * jacobianPart.transpose() * d * jacobianPart * protoM1;
    }

    void assembleGlobalStiffnessMatrix()
    {
        K.reserve(elements.size() * degreesOfFreedom); // #elements * degreesOfFreedom
        for (int elNum = 0; elNum < elements.size(); elNum++)
        {
            const std::vector<int>& element = elements[elNum];
            DenseMatrix localM1 = calculateLocalStiffnessMatrix(element, elNum);
            for (int i = 0; i < degreesOfFreedom; i++)
            {
                const std::pair<double, double>& dot0 = nodes[element[0]];
                const std::pair<double, double>& dot1 = nodes[element[1]];
                const std::pair<double, double>& dot2 = nodes[element[2]];
                double J = dot0.first * dot1.second + dot0.second * dot2.first + dot1.first * dot2.second
                    - dot0.first * dot2.second - dot0.second * dot1.first - dot2.first * dot1.second;
                for (int j = 0; j < degreesOfFreedom; j++)
                {
                    for (int vki = 0; vki < variablesCount; vki++)
                    {
                        for (int vkj = 0; vkj < variablesCount; vkj++)
                        {
                            K.emplace_back(2 * element[j] + vkj, 2 * element[i] + vki, abs(J) / 2 * localM1(2 * j + vkj, 2 * i + vki));// division of 2 because of integrating constants over the reference triangle
                        }
                    }
                }
            }
        }
        std::cout << "Assembled global stiffness matrix!\n";
    }

    void assembleGlobalMassMatrix()
    {
        M.reserve(elements.size() * degreesOfFreedom); // #elements * degreesOfFreedom
        for (int elementNum = 0; elementNum < elements.size(); elementNum++)
        {
            const std::vector<int>& element = elements[elementNum];
            for (int i = 0; i < degreesOfFreedom; i++)
            {
                for (int j = 0; j < degreesOfFreedom; j++)
                {
                    for (int vki = 0; vki < variablesCount; vki++)
                    {
                        for (int vkj = 0; vkj < variablesCount; vkj++)
                        {
                            const std::pair<double, double>& dot0 = nodes[element[0]];
                            const std::pair<double, double>& dot1 = nodes[element[1]];
                            const std::pair<double, double>& dot2 = nodes[element[2]];
                            double J = dot0.first * dot1.second + dot0.second * dot2.first + dot1.first * dot2.second
                                - dot0.first * dot2.second - dot0.second * dot1.first - dot2.first * dot1.second;
                            M.emplace_back(2 * element[j] + vkj, 2 * element[i] + vki, abs(J) * getRho(element, elementNum) * localM0(2 * j + vkj, 2 * i + vki));
                        }
                    }
                }
            }
        }
        std::cout << "Assembled global mass matrix!\n";
    }

    std::pair<double, double> initialConditionsAnalyticalModel(double x, double y) const
    {
        // Compressive wave
        //double ux = -2 * x / (pow((x*x + 1), 2) * (y * y + 1));
        //double uy = -2 * y / (pow((y * y + 1), 2) * (x * x + 1));
        // P and S waves
        double a = 1;
        double ux = -2* x * std::exp(-(x * x + y * y) / pow(a, 2))/pow(a,2);
        double uy = - 2*y * std::exp(-(x * x + y * y) / pow(a, 2))/pow(a,2)*0;
        return { ux, uy };
    }
    void assembleGlobalVectorByNodes()
    {
        for (int i = 0; i < localVector.size(); i++)
        {
            if (localVector[i] != 0)
            {
                V.emplace_back(i, 0, localVector[i]);
            }
        }
    }
    void assembleGlobalVector(const std::vector<int>& sourceIndices)
    {
        if (modelProblem == "Analytical")
        {
            assembleGlobalVectorByNodes();
            return;
        }
        std::cout << "Wiill assemble global vector";
        V.reserve(elements.size() * degreesOfFreedom); // #elements * degreesOfFreedom
        // if source is space-dependant we have to use (custom) local matrix here as well as calculateSourceVx and calculateSourceVy
        // the vector will have to be computed for every time layer- either with analytical formula(integrated on hand) or numerical integration
        std::vector<double> integratedPsi = { 1./6., 1./6, 1./6 };
        for (int sourceIndex : sourceIndices)
        {
            const std::vector<int>& element = elements[sourceIndex];
            for (int i = 0; i < degreesOfFreedom; i++)
            {
                for (int vk = 0; vk < variablesCount; vk++)
                {
                    const std::pair<double, double>& dot0 = nodes[element[0]];
                    const std::pair<double, double>& dot1 = nodes[element[1]];
                    const std::pair<double, double>& dot2 = nodes[element[2]];
                    double J = dot0.first * dot1.second + dot0.second * dot2.first + dot1.first * dot2.second
                        - dot0.first * dot2.second - dot0.second * dot1.first - dot2.first * dot1.second;
                    const int currentIndex = variablesCount * i + vk;
                    V.emplace_back(variablesCount * element[i] + vk, 0, abs(J));
                }
            }
        }
        std::cout << "Assembled global vector";
    }

    //newer version
    void assembleGlobalVectorFromNodes(const std::vector<int>& sourceIndices)
    {
        if (modelProblem == "Analytical")
        {
            assembleGlobalVectorByNodes();
            return;
        }
        std::cout << "Wiill assemble global vector";
        V.reserve(nodes.size() * degreesOfFreedom); // #elements * degreesOfFreedom
        std::cout << "Source indices are " << sourceIndices.size() << "\n";
        for (int sourceIndex : sourceIndices)
        {
                for (int vk = 0; vk < variablesCount; vk++)
                {
                    V.emplace_back(variablesCount * sourceIndex + vk, 0, 1);
                }
        }
        std::cout << "Assembled global vector";
    }

};
