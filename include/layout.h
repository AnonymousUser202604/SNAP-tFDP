#ifndef LAYOUT_H
#define LAYOUT_H

#include "graph.h"

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

    void _run_ours(const LayoutConfig& config) const;
    void _run_ours_parallel(const LayoutConfig& config);

    void _run_ours_gpu(const LayoutConfig& config);

public:
    explicit Layout(Graph& g)
        : g(g), N(g.N), M(g.M), Y(g.Y),
          edges(g.edges), degree(g.degree)
    {
    }

    void run(const LayoutConfig& config)
    {
        if (config.gpu)
            _run_ours_gpu(config);
        else if (config.parallel)
            _run_ours_parallel(config);
        else
            _run_ours(config);
    }
};


#endif //LAYOUT_H
