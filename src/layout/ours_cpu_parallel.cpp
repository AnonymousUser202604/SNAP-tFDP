#include "layout.h"

#include <cassert>
#include <iostream>
#include <omp.h>
#include <queue>

#ifdef PARALLEL_SORT
#ifdef __GNUC__
#include <parallel/algorithm>
#else
#include <algorithm>
#include <execution>
#endif
#endif

#include "pcg_random.hpp"
#include "utils.h"

void Layout::_run_ours_parallel(const LayoutConfig& config)
{
    int neg_num = config.k;
    int random_seed = config.random_seed;
    int n_epoch = config.n_epoch;

    short n_threads = config.n_threads;
    if (n_threads < 1 || n_threads > omp_get_max_threads())
    {
        std::cout << "warning: param `n_threads` not set properly. "
            << "got: `n_threads = " << n_threads << "`. "
            << "automatically setting n_threads = " << (n_threads = omp_get_max_threads()) << std::endl;
    }
    omp_set_num_threads(n_threads);

    std::vector<short> node_thread(N);
    std::vector<std::pair<size_t, size_t>> thread_edges_range_index(n_threads);
    std::vector<size_t> thread_nodes_size(n_threads);

    Timer t_sort("partition and sort");
    // step.1 partition
    {
        struct PT
        {
            short t_id; // thread id
            int64_t size; // current number of edges in this thread

            // override operator`<` intending a min heap
            bool operator <(const PT& rv) const
            {
                return size > rv.size;
            }
        };

        std::priority_queue<PT> pq;
        for (short i = 0; i < n_threads; ++i)
        {
            pq.push(PT{.t_id = i, .size = 0});
        }

        std::vector<int> idx(N);
        for (int i = 0; i < N; ++i)
            idx[i] = i;
        std::sort(idx.begin(), idx.end(),
                  [&](const int& a, const int& b)
                  {
                      return degree[a] > degree[b];
                  }
        );

        for (const int& node_id : idx)
        {
            const auto [t_id,size] = pq.top();
            pq.pop();
            node_thread[node_id] = t_id;
            thread_nodes_size[t_id]++;
            pq.push(PT{t_id, size + degree[node_id]});
        }

#ifdef PARALLEL_SORT
        // speed up sorting, taking O(M) additional mem.
#ifdef __GNUC__
        __gnu_parallel::sort(edges.begin(), edges.end(),
                             [&](const auto& a, const auto& b)
                             {
                                 return node_thread[a.first] < node_thread[b.first];
                             }
        );
#else
        std::sort(std::execution::par, edges.begin(), edges.end(),
                  [&](const auto& a, const auto& b)
                  {
                      return node_thread[a.first] < node_thread[b.first];
                  }
        );
#endif
#else
        std::sort(edges.begin(), edges.end(),
                  [&](const auto& a, const auto& b)
                  {
                      return node_thread[a.first] < node_thread[b.first];
                  }
        );
#endif
        t_sort.stop();

        int64_t sum_size = 0;
        std::vector<int64_t> thread_edges_size(n_threads);
        while (!pq.empty())
        {
            const auto [t_id,size] = pq.top();
            pq.pop();
#ifdef VERBOSE
            printf("%lld:%.3lf  ", t_id, static_cast<double>(size) / M);
#endif
            thread_edges_size[t_id] = size;
            sum_size += size;
        }
        printf("\nsum of pairs:%lld\n", sum_size);
        assert(sum_size==M);

        thread_edges_range_index[0] = std::make_pair(0ll, thread_edges_size[0]);
        for (int i = 1; i < n_threads; ++i)
        {
            const auto [l,r] = thread_edges_range_index[i - 1];
            thread_edges_range_index[i] = std::make_pair(r, r + thread_edges_size[i]);
        }
#ifdef VERBOSE
        for (int i = 0; i < n_threads; ++i)
        {
            const auto [l,r] = thread_edges_range_index[i];
            printf("t_%.2d: %lld %lld size=%lld\n", i, l, r, r - l);
        }
        fflush(stdout);
#endif
    }


    // step.2 shuffle edges array in each thread
    {
        Timer t_shuffle("shuffle");
        for (int i = 0; i < n_threads; ++i)
        {
            std::mt19937 rng(random_seed + i);
            const auto [l,r] = thread_edges_range_index[i];
            std::shuffle(edges.begin() + l, edges.begin() + r, rng);
        }
        t_shuffle.stop();

        for (int i = 0; i < n_threads; ++i)
        {
            const auto [l,r] = thread_edges_range_index[i];
            assert(node_thread[edges[l].first]==i);
            assert(node_thread[edges[(l+r-1)/2].first]==i);
            assert(node_thread[edges[r-1].first]==i);
        }
    }


    // step.3 iteratively update
    // NOTE:
    // Each edge (u, v) only updates u to avoid cross-thread write conflicts.
    // This implements an asymmetric, lock-free SGD variant.
    Timer t_layout("layout");
    auto& edges = this->edges;
    auto Y = this->Y;
    auto N = this->N;
    auto M = this->M;

#ifdef VERBOSE
    std::vector<int> sampled_node(config.neg * M);
#endif

#pragma omp parallel
    {
        const int t_id = omp_get_thread_num();
        const auto& [edges_begin,edges_end] = thread_edges_range_index[t_id];

        pcg64 pcg(random_seed + t_id);
        std::uniform_int_distribution<int> gen_node_n_2(0, N - 2);

        double step = 1.0;
        const double stepMin = 0.01;
        const double stepSize = 1.0 - pow(0.02, 1.0 / n_epoch);

        for (int epoch = 0; epoch < n_epoch; ++epoch)
        {
            double dx, dy, l, mvx, mvy;
            for (auto i = edges_begin; i < edges_end; ++i)
            {
                const auto [u,v] = edges[i];

                dx = Y[v * 2] - Y[u * 2];
                dy = Y[v * 2 + 1] - Y[u * 2 + 1];

                l = 0.1 + 0.8 / (1 + dx * dx + dy * dy);

                mvx = step * l * dx;
                mvy = step * l * dy;

                Y[u * 2] += mvx;
                Y[u * 2 + 1] += mvy;
                Y[v * 2] -= mvx;
                Y[v * 2 + 1] -= mvy;

                for (int j = 0; j < neg_num; ++j)
                {
                    const int _id = gen_node_n_2(pcg);
                    const int other_id = _id + (_id >= u);
#ifdef VERBOSE
                    if (epoch == n_epoch - 1)
                        sampled_node[i * neg_num + j] = other_id;
#endif


                    dx = Y[other_id * 2] - Y[u * 2];
                    dy = Y[other_id * 2 + 1] - Y[u * 2 + 1];

                    l = -1.0 / ((1 + dx * dx + dy * dy) * (1 + dx * dx + dy * dy));

                    mvx = step * l * dx;
                    mvy = step * l * dy;

                    Y[u * 2] += mvx;
                    Y[u * 2 + 1] += mvy;
                    Y[other_id * 2] -= mvx;
                    Y[other_id * 2 + 1] -= mvy;
                }
            }
            step += (stepMin - step) * stepSize;
#ifdef VERBOSE
#pragma omp single
            {
                printf("t_id(%d): epoch(%d)\n", t_id, epoch);
                if (epoch == n_epoch - 1)
                {
                    double cur_actual_loss_attr = 0., cur_actual_loss_rep = 0.;
                    for (int i = 0; i < M; ++i)
                    {
                        const auto& [u,v] = edges[i];
                        double dx, dy, dis2;
                        dx = Y[v * 2] - Y[u * 2];
                        dy = Y[v * 2 + 1] - Y[u * 2 + 1];
                        dis2 = dx * dx + dy * dy;
                        cur_actual_loss_attr += 0.05 * dis2 + 0.4 * std::log(1 + dis2);
                        for (int j = 0; j < neg_num; ++j)
                        {
                            const auto& w = sampled_node[i * neg_num + j];
                            dx = Y[w * 2] - Y[u * 2];
                            dy = Y[w * 2 + 1] - Y[u * 2 + 1];
                            dis2 = dx * dx + dy * dy;
                            cur_actual_loss_rep += 0.5 / (1 + dis2);
                        }
                    }
                    FILE* f = fopen(config.fp_loss.c_str(), "w");
                    fprintf(f, "epoch,actual_loss_attr,actual_loss_rep,actual_loss_total\n");
                    fprintf(f, ""
                            "%d,"
                            "%lf,%lf,%lf\n",
                            config.n_epoch,
                            cur_actual_loss_attr, cur_actual_loss_rep, cur_actual_loss_attr + cur_actual_loss_rep
                    );
                    fclose(f);
                }
            }
#else
#pragma omp single
            {
                printf("t_id(%d): epoch(%d)\n", t_id, epoch);
            }
#endif
        }
    }
    t_layout.stop();
}
