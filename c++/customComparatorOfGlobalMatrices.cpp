#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <array>
#include <chrono>

struct Point {
    double x, y;
};

struct Element {
    int a, b, c;
};

constexpr double TOL = 1e-1asa10;

//--------------------------------------
// Read CSV utilities
//--------------------------------------

std::vector<Point> readNodes(const std::string& filename) {
    std::vector<Point> nodes;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string sx, sy;
        std::getline(ss, sx, ',');
        std::getline(ss, sy, ',');
        nodes.push_back({ std::stod(sx), std::stod(sy) });
    }
    return nodes;
}

std::vector<Element> readElements(const std::string& filename) {
    std::vector<Element> elems;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string sa, sb, sc;
        std::getline(ss, sa, ',');
        std::getline(ss, sb, ',');
        std::getline(ss, sc, ',');
        elems.push_back({ std::stoi(sa), std::stoi(sb), std::stoi(sc) });
    }
    return elems;
}

//--------------------------------------
// Tolerance-aware point check
//--------------------------------------

int findOrAddPoint(const Point& p, std::vector<Point>& points, double tol = TOL) {
    for (int i = 0; i < (int)points.size(); ++i) {
        double dx = points[i].x - p.x;
        double dy = points[i].y - p.y;
        if (std::fabs(dx) < tol && std::fabs(dy) < tol) {
               return i;
        }
    }
    points.push_back(p);
    return points.size() - 1;
}

//--------------------------------------
// Refinement
//--------------------------------------

std::vector<Element> refineMesh(std::vector<Point>& points, const std::vector<Element>& elements, int sourceElementIndex, double tol = TOL) {
    std::vector<Element> refined;

    for (int i = 0; i < elements.size(); i++)
    {
        std::cout << i;
        const auto& e = elements[i];
        Point p1 = points[e.a];
        Point p2 = points[e.b];
        Point p3 = points[e.c];

        Point m1 = { (p1.x + p2.x) * 0.5, (p1.y + p2.y) * 0.5 };
        Point m2 = { (p2.x + p3.x) * 0.5, (p2.y + p3.y) * 0.5 };
        Point m3 = { (p3.x + p1.x) * 0.5, (p3.y + p1.y) * 0.5 };

        int i1 = findOrAddPoint(m1, points, tol);
        int i2 = findOrAddPoint(m2, points, tol);
        int i3 = findOrAddPoint(m3, points, tol);

        if (i == sourceElementIndex)
        {
            std::cout << "Refining a source element! New indices will be " << refined.size() << ", " << refined.size() + 1 << ", " << refined.size() + 2 << ", " << refined.size() + 3;
        }
        refined.push_back({ e.a, i1, i3 });
        refined.push_back({ i1, e.b, i2 });
        refined.push_back({ i3, i2, e.c });
        refined.push_back({ i1, i2, i3 });
    }

    return refined;
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    int sourceElementIndex = 1602;
    std::string configPath = "D:\\Serious\\FMI\\Магистър\\Дипломен семестър\\FEM2D_config";
    std::string suffix = ".csv";

    std::vector<Point> points = readNodes(configPath + "\\nodes.csv");
    std::vector<Element> elements = readElements(configPath + "\\elements.csv");

    std::cout << "Initial points: " << points.size() << "\n";
    std::cout << "Initial elements: " << elements.size() << "\n";

    std::vector<Element> refined = refineMesh(points, elements, sourceElementIndex);

    std::cout << "Refined points: " << points.size() << "\n";
    std::cout << "Refined elements: " << refined.size() << "\n";

    // Write back to CSV
    std::ofstream outNodes(configPath + "\\nodes" + suffix);
    for (auto& p : points)
        outNodes << p.x << "," << p.y << "\n";

    std::ofstream outElems(configPath + "\\elements" + suffix);
    for (auto& e : refined)
        outElems << e.a << "," << e.b << "," << e.c << "\n";

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Done in \n" << elapsed.count() << "seconds.";
    return 0;
}