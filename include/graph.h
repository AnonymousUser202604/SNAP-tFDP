#ifndef GRAPH_H
#define GRAPH_H

#include <random>
#include <vector>
#include <string>
#include <algorithm>

class Graph
{
public:
    Graph() : N(0), M(0), Y(nullptr)
    {
    }

    ~Graph() { delete[] Y; }
    void readGraph(std::string filename);
    void init_PMDS(std::string filename);
    void init_random(int random_seed);
    void init_spiral();

    void normalizePos() const;
    void savePos(std::string filename);
    void drawSvg(std::string attrFilename, std::string svgFilename);

    std::pair<double, double> calc_loss(int k);

    size_t N, M;
    double* Y{nullptr};
    std::vector<std::pair<int, int>> edges;
    std::vector<int> degree;
};


#endif
