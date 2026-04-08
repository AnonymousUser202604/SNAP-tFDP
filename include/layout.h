#ifndef LAYOUT_H
#define LAYOUT_H

#include "graph.h"
#include <iostream>

inline constexpr double EPS = 1e-5;

struct LayoutConfig
{
    int n_epoch{300};
    int k{5};

    int random_seed{42};

    bool parallel{false};
    bool gpu{false};
    int n_threads{0};
};

class Layout
{
protected:
    // source graph and unpacked fields
    Graph& g;
    size_t N, M;
    double* Y;
    std::vector<std::pair<int, int>>& edges;
    const std::vector<int>& degree;

    void run_serial(const LayoutConfig& config) const;
    void run_parallel(const LayoutConfig& config);
    void run_gpu(const LayoutConfig& config);

public:
    explicit Layout(Graph& g)
        : g(g), N(g.N), M(g.M), Y(g.Y),
          edges(g.edges), degree(g.degree)
    {
    }

    void run(const LayoutConfig& config)
    {
        if (config.gpu)
        {
#ifdef ENABLE_CUDA
            run_gpu(config);
#else
            std::cerr << "Error: ENABLE_CUDA option was not set." << std::endl;
            exit(-1);
#endif
        }
        else if (config.parallel)
        {
#ifdef ENABLE_PARALLEL
            run_parallel(config);
#else
            std::cerr << "Error: ENABLE_PARALLEL option was not set." << std::endl;
            exit(-1);
#endif
        }
        else
            run_serial(config);
    }
};


#endif //LAYOUT_H
