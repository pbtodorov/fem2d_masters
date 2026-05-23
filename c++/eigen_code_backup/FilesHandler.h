#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <unordered_map>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include "ConfigParser.h"

#pragma once
using T = Eigen::Triplet<double>;
using TripletsList = std::vector<T>;


class FilesHandler
{
public:
    std::string configPath;
    std::vector<std::pair<double, double>> nodes{};
    std::vector<std::vector<int>> elements{};
    std::vector<int> fakeNodes;

    FilesHandler(const std::string& config) :
        configPath(config)
    {
        readElementsCSV();
        readNodesCSV();
        readConfig();
        // Deprecated- no longer using mfem for this
        //getFakeBoundary();
    }

    // Deprecated- no longer using mfem for this
    void getFakeBoundary()
    {
        std::string filename = configPath + "\\nonFreeSurfaceNodes";
        std::ifstream file(filename);
        std::string line;
        std::getline(file, line);
        std::vector<int> tempFakeNodes = getIndices(line);
        for (int fakeNode : tempFakeNodes)
        {
            fakeNodes.push_back(2 * nodes.size() + 2 * fakeNode);
            fakeNodes.push_back(2 * nodes.size() + 2 * fakeNode + 1);
        }
    }

    void readElementsCSV()
    {
        std::cout << "Reading elements...\n";
        std::string filename = configPath + "\\elements.csv";
        std::ifstream file(filename);
        std::string line;

        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
        }

        while (std::getline(file, line)) {
            std::vector<int> row;
            std::stringstream ss(line);
            std::string cell;

            while (std::getline(ss, cell, ',')) {
                row.push_back(std::stoi(cell));
            }

            elements.push_back(row);
        }

        file.close();
    }

    void readNodesCSV() 
    {
        std::cout << "Reading nodes...\n";
        std::string filename = configPath + "\\nodes.csv";
        std::ifstream file(filename);
        std::string line;

        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
        }

        while (std::getline(file, line)) {
            std::pair<double, double> row;
            std::stringstream ss(line);
            std::string cell;

            std::getline(ss, cell, ',');
            row.first = std::stod(cell);

            std::getline(ss, cell, ',');
            row.second = std::stod(cell);

            nodes.push_back(row);
        }

        file.close();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void saveSparseMatrixInFileMFEMStyle(const std::string& filename, const TripletsList& TripletsList, bool isVector = false/*,
                                         int halfSize, const std::function<int(int, int)>& comparator*/) const
    {
        std::ofstream matrixStream(configPath + "\\Eigen\\" + filename);
        int prevRow = 0;
        int prevCol = 0;
        double currentIndexCombined = 0;
        for (const T& trip : TripletsList)
        {
            if (trip.value() == 0)
                continue;
            int row = trip.row();//comparator(trip.row(), halfSize);
            int col = trip.col();// comparator(trip.col(), halfSize);
            if (row != prevRow || col != prevCol)
            {
                matrixStream << prevRow << ", " << prevCol << ", " << currentIndexCombined << "\n";
                currentIndexCombined = 0;
            }
            while (row > prevRow)
            {
                prevRow += 1;
                if (!isVector)
                {
                    matrixStream << std::endl;
                }
            }
            currentIndexCombined += trip.value();

            prevCol = col;
        }
        matrixStream << prevRow << ", " << prevCol << ", " << currentIndexCombined << "\n";
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void readSparseMatrixFromFile(const std::string& filename, TripletsList& triplets) const {
        std::ifstream infile(filename);
        if (!infile) {
            throw std::runtime_error("Could not open file: " + filename);
        }

        // First pass: determine matrix dimensions and collect triplets
        int max_row = 0;
        int max_col = 0;
        std::string line;

        while (std::getline(infile, line)) {
            // Skip empty lines
            if (line.empty()) {
                continue;
            }
            size_t first_comma = line.find(", ");
            if (first_comma == std::string::npos) {
                std::cerr << "Warning: Skipping malformed line (missing first comma): " << line << std::endl;
                continue;
            }

            size_t second_comma = line.find(", ", first_comma + 2);
            if (second_comma == std::string::npos) {
                std::cerr << "Warning: Skipping malformed line (missing second comma): " << line << std::endl;
                continue;
            }

            int row, col;
            double value;

            try {
                row = std::stoi(line.substr(0, first_comma));
                col = std::stoi(line.substr(first_comma + 2, second_comma - first_comma - 2));
                value = std::stod(line.substr(second_comma + 2));
            }
            catch (const std::exception& e) {
                std::cerr << "Warning: Error parsing line (invalid numbers): " << line << std::endl;
                continue;
            }

            // Update maximum dimensions
            max_row = std::max(max_row, row + 1);  // +1 because indices are 0-based
            max_col = std::max(max_col, col + 1);  // +1 because indices are 0-based

            triplets.emplace_back(row, col, value);
        }
    }

    void readSparseVectorFromFile(const std::string& filename, TripletsList& triplets) const {
        std::ifstream infile(filename);
        if (!infile) {
            throw std::runtime_error("Could not open file: " + filename);
        }

        // First pass: determine matrix dimensions and collect triplets
        int max_row = 0;
        std::string line;

        while (std::getline(infile, line)) {
            // Skip empty lines
            if (line.empty()) {
                continue;
            }

            size_t first_comma = line.find(", ");
            if (first_comma == std::string::npos) {
                std::cerr << "Warning: Skipping malformed line (missing first comma): " << line << std::endl;
                continue;
            }

            int row;
            double value;

            try {
                row = std::stoi(line.substr(0, first_comma));
                value = std::stod(line.substr(first_comma + 2));
            }
            catch (const std::exception& e) {
                std::cerr << "Warning: Error parsing line (invalid numbers): " << line << std::endl;
                continue;
            }

            // Update maximum dimensions
            max_row = std::max(max_row, row + 1);  // +1 because indices are 0-based

            triplets.emplace_back(row, 0, value);
        }
    }

    void readMFEMGlobalMatrices(TripletsList& m0, TripletsList& m1, TripletsList& vec) const
    {
        m0.clear();
        m1.clear();
        const std::string& m0Path = configPath + "\\MFEM\\" + "globalMassMFEM";
        readSparseMatrixFromFile(m0Path, m0);
        const std::string& m1Path = configPath + "\\MFEM\\" + "globalStiffMFEM";
        readSparseMatrixFromFile(m1Path, m1);
        const std::string& vPath = configPath + "\\MFEM\\" + "globalVectorMFEM";
        readSparseVectorFromFile(vPath, vec);
    }
    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    std::unordered_map<std::string, std::string> readConfig() const {
        std::cout << "Reading config file...";
        std::ifstream file(configPath + "\\config.txt");
        std::string line;
        std::unordered_map<std::string, std::string> configTxt;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string key, value;
            if (std::getline(iss, key, '=') && std::getline(iss, value)) {
                configTxt[key] = value;
            }
        }
        return configTxt;
    }

};

