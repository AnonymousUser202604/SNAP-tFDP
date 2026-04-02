#include <chrono>

#include "layout.h"

#include "pcg_random.hpp"

void Layout::_run_ours(const LayoutConfig& config) const
{
    // 用于负采样的随机数生成器
    pcg64 pcg(config.random_seed);
    std::uniform_int_distribution<int> gen_node_n_2(0, N - 2);
    std::uniform_int_distribution<int> gen_node_n_1(0, N - 1);
    // 不采样当前点本身。采样点: g=gen_node(pcg) other_id=g+(g>=u)


    // 打乱边的顺序
    std::mt19937 rng(config.random_seed); // 使用 Mersenne Twister 作为随机数引擎
    shuffle(edges.begin(), edges.end(), rng); // Fisher-Yates Shuffle (shuffle 实现) 打乱


    // 缓存数据
    constexpr int CACHE_SIZE = 1 << 16;
    static int cached_node_ids[CACHE_SIZE];
    static double cached_node_Y[CACHE_SIZE * 2];
    static uint16_t ptr_cache = 0;


    // 温度策略
    double step = 1.0;
    double stepMin = 0.01;
    double stepSize = 1.0 - pow(0.02, 1.0 / config.n_epoch);

    // tfdp参数
    const double alpha = 0.1, beta = 8;

#ifdef VERBOSE
    std::vector<int> sampled_node(config.k * M);
#endif

    const auto time_start = std::chrono::high_resolution_clock::now();
    for (int epoch = 0; epoch < config.n_epoch; ++epoch)
    {
        for (size_t i = 0; i < M; ++i)
        {
            const auto& [u,v] = edges[i];

            // 引力
            double dx = Y[v * 2] - Y[u * 2];
            double dy = Y[v * 2 + 1] - Y[u * 2 + 1];
            double dis2 = dx * dx + dy * dy;

            double l = alpha * (1 + beta / (1 + dis2));

            double mvx = step * l * dx;
            double mvy = step * l * dy;

            Y[u * 2] += mvx;
            Y[u * 2 + 1] += mvy;
            Y[v * 2] -= mvx;
            Y[v * 2 + 1] -= mvy;

            // 斥力
            for (int j = 0; j < config.k; ++j)
            {
                int other_id{-1};
                int _id = gen_node_n_1(pcg);
                while (_id == u)
                    _id = gen_node_n_1(pcg);
                other_id = _id;

                dx = Y[other_id * 2] - Y[u * 2];
                dy = Y[other_id * 2 + 1] - Y[u * 2 + 1];

                dis2 = dx * dx + dy * dy;
                l = -1.0 / ((1 + dis2) * (1 + dis2));

                mvx = step * l * dx;
                mvy = step * l * dy;

                Y[u * 2] += mvx;
                Y[u * 2 + 1] += mvy;
                Y[other_id * 2] -= mvx;
                Y[other_id * 2 + 1] -= mvy;
#ifdef VERBOSE
                {
                    sampled_node[i * config.k + j] = other_id;
                }
#endif
            }
        }
        step += (stepMin - step) * stepSize;

#ifdef VERBOSE
        if (epoch == config.n_epoch - 1)
        {
            double cur_actual_loss_attr = 0., cur_actual_loss_rep = 0.;
            cur_actual_loss_rep = cur_actual_loss_attr = 0.;
            for (size_t i = 0; i < M; ++i)
            {
                const auto& [u,v] = edges[i];
                double dx = Y[v * 2] - Y[u * 2];
                double dy = Y[v * 2 + 1] - Y[u * 2 + 1];
                double dis2 = dx * dx + dy * dy;
                cur_actual_loss_attr += 0.05 * dis2 + 0.4 * std::log(1 + dis2);
                for (int j = 0; j < config.k; ++j)
                {
                    const auto& w = sampled_node[i * config.k + j];
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
        printf("epoch: %d\n", epoch);
#endif
    }
    const auto time_end = std::chrono::high_resolution_clock::now();
    const double time_s =
        std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e6;
    printf("time used: %.3lfs\n", time_s);
}
