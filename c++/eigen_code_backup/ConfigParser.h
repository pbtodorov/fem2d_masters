#pragma once
#include <unordered_map>
#include <string>

std::vector<int> getIndices(const std::string& indicesLabel);
std::vector<double> getDoubles(const std::string& indicesStr);

class ConfigParser{
public:
	bool useMFEMMatrices;
	double tau;
    double Tmax;
    bool debug;
    int allNodes;
    std::string mcm;
	// element indices of the source
	std::vector<int> sourceIndices;
	// nodes indices on which we save the output
	std::vector<int> seismographIndices;
    std::vector<int> fakeBoundary;
    std::vector<double> depths;
    double left;
    double right;
    std::string cavityFilling;
    std::string modelProblem;
    std::vector<int> cavityIndices;
    // one or more epicentres
    std::vector<double> sourceEpicentre;
    double sourceRadius;

	ConfigParser(std::unordered_map<std::string, std::string> configTxt, int dim)
	{
        useMFEMMatrices = std::stoi(configTxt["useMfemMatrices"])==1;
        allNodes = std::stoi(configTxt["allNodes"]);
        debug = std::stoi(configTxt["debug"])==1;
        mcm = configTxt["mcm"];
		tau = std::stod(configTxt["tau"]);
        Tmax = std::stod(configTxt["Tmax"]);
        cavityFilling = configTxt["cavityFilling"];
        if (!cavityFilling.empty() && cavityFilling != "a" && cavityFilling != "w" && cavityFilling != "m")
        {
            std::cout << "Unexpected option value passed for cavityFilling. Allowed values are 'a' and 'w'!";
            throw std::runtime_error("Unexpected option value passed for cavityFilling");
        }
        cavityIndices = getIndices(configTxt["cavityIndices"]);

        modelProblem = configTxt["modelProblem"];
        if (modelProblem != "Halfspace" && modelProblem != "Layered" && modelProblem != "Elastic" && modelProblem != "Analytical")
        {
            std::cout << "Unexpected option value passed for modelProblem. Allowed values are 'Halfspace', 'Layered', 'Elastic' or 'Analytical'!";
            throw std::runtime_error("Unexpected option value passed for modelProblem");
        }
        depths = getDoubles(configTxt["depths"]);
        left = std::stod(configTxt["left"]);
        right = std::stod(configTxt["right"]);

        sourceIndices = getIndices(configTxt["sourceElement"]);

        //sourceEpicentre = getDoubles(configTxt["sourceEpicentre"]);
        //sourceRadius = std::stod(configTxt["sourceRadius"]);

        std::vector<int> tempSeismographIndices = getIndices(configTxt["seismographIndices"]);
        // eigen format
        for (int index : tempSeismographIndices)
        {
            seismographIndices.emplace_back(dim + 2 * index);
            seismographIndices.emplace_back(dim + 2 * index + 1);
            seismographIndices.emplace_back(2 * index);
            seismographIndices.emplace_back(2 * index + 1);
        }
	}

    std::string createOutputFilename(const std::string& configPath, bool isSnapshot) const
    {
        std::string outputFile = configPath + "\\results\\";

        if (useMFEMMatrices)
            outputFile += "MFEM_";
        else
        {
            outputFile += "Eigen_";
        }

        outputFile += "seismogram_";

        if (isSnapshot)
        {
            outputFile += "snapshots_";
        }
        else if (allNodes > 0)
        {
            outputFile += "all_points_";
        }
        else
        {
            outputFile += "displ_vel_";
        }

        outputFile += mcm;
        outputFile += ".csv";
        std::cout << "Will write in " << outputFile << '\n';
        return outputFile;
    }
    
};

std::vector<int> getIndices(const std::string& indicesStr)
{
    std::vector<int> res;
    if (indicesStr.empty())
        return res;

    size_t start = 0;
    size_t end;
    const std::string delimiter = ", ";
    while ((end = indicesStr.find(delimiter, start)) != std::string::npos) {
        res.push_back(stoi(indicesStr.substr(start, end - start)));
        start = end + delimiter.length();
    }
    res.push_back(stoi(indicesStr.substr(start)));

    return res;
}

std::vector<double> getDoubles(const std::string& indicesStr)
{
    std::vector<double> res;

    size_t start = 0;
    size_t end;
    const std::string delimiter = ", ";
    while ((end = indicesStr.find(delimiter, start)) != std::string::npos) {
        res.push_back(stod(indicesStr.substr(start, end - start)));
        start = end + delimiter.length();
    }
    res.push_back(stod(indicesStr.substr(start)));

    return res;
}