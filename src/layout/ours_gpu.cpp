#include "layout.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <omp.h>

// Initialize CUDA runtime once at program startup
static void cuda_warming_up()
{
    float* dummy = nullptr;
    cudaMalloc(&dummy, sizeof(float));
    cudaFree(dummy);
}

extern "C" void launch_ours_gpu_kernel(
    const int* d_edge,
    float* d_Y,
    uint64_t N,
    uint32_t n_threads,
    uint64_t M,
    int neg_num,
    float step,
    unsigned int seed
#ifdef VERBOSE
    ,
    int *d_sampled_node,
    int is_last_epoch
#endif
);
#endif

void Layout::_run_ours_gpu(const LayoutConfig& config)
{
#ifndef ENABLE_CUDA
    std::cerr << "Error: GPU support not compiled. Rebuild with -DENABLE_CUDA=ON" << std::endl;
    return;
#else
    cuda_warming_up();

    const int& neg_num = config.k;
    const int& random_seed = config.random_seed;
    const int& n_epoch = config.n_epoch;

    // ================================================================
    // Step 1: Determine number of threads
    // ================================================================
    int n_threads = config.n_threads;
    if (n_threads < 1)
    {
        n_threads = 4;
        for (int i = 8; i <= (1 << 13); i <<= 1)
            if (abs(M / i - 1000) < abs(M / n_threads - 1000))
                n_threads = i;
        std::cout << "warning: param `n_threads` not set properly. "
            << "automatically setting n_threads = " << n_threads << std::endl;
    }
    // ================================================================
    // Step 2: Shuffle all edges randomly
    // ================================================================
    Timer t_partition("Shuffle");
    {
        std::mt19937 rng(random_seed);
        std::shuffle(edges.begin(), edges.end(), rng);
    }
    t_partition.stop();

    // ================================================================
    // Step 5: Reorganize edges into interleaved format early (for later loss calculation)
    // ================================================================
    std::vector<int> edge_data(M * 2);
    for (int i = 0; i < M; ++i)
    {
        edge_data[i * 2] = edges[i].first;
        edge_data[i * 2 + 1] = edges[i].second;
    }

    // ================================================================
    // Step 6: Allocate GPU memory and copy data
    // ================================================================
    Timer t_alloc("GPU memory alloc");
    int* d_edge = nullptr;
    float* d_Y = nullptr;
    int* d_sampled_node = nullptr;

    cudaMalloc(&d_edge, M * 2 * sizeof(int));
    cudaMalloc(&d_Y, N * 2 * sizeof(float));

#ifdef VERBOSE
    if (!config.fp_loss.empty())
    {
        cudaMalloc(&d_sampled_node, config.neg * M * sizeof(int));
    }
#endif

    if (!d_edge || !d_Y)
    {
        std::cerr << "Error: GPU memory allocation failed" << std::endl;
        cudaFree(d_edge);
        cudaFree(d_Y);
        cudaFree(d_sampled_node);
        return;
    }
    t_alloc.stop();

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    double used = (total_mem - free_mem) / 1024.0 / 1024.0;

    printf("[GPU Peak] %.2f MB\n", used);

    Timer t_copy("host -> device copy");

    cudaMemcpy(d_edge, edge_data.data(), M * 2 * sizeof(int), cudaMemcpyHostToDevice);

    // Convert double Y to float for GPU
    std::vector<float> Y_float(N * 2);
    for (int i = 0; i < N * 2; ++i)
        Y_float[i] = static_cast<float>(Y[i]);
    cudaMemcpy(d_Y, Y_float.data(), N * 2 * sizeof(float), cudaMemcpyHostToDevice);
    t_copy.stop();

    // ================================================================
    // Step 7: Run epochs (step decay aligned with CPU parallel version)
    // ================================================================
    float step = 1.0f;
    const float stepMin = 0.01f;
    const float stepSize = 1.0f - powf(0.02f, 1.0f / n_epoch);

    Timer t_layout("GPU layout");
    for (int epoch = 0; epoch < n_epoch; ++epoch)
    {
        const unsigned int seed = random_seed + epoch;
#ifdef VERBOSE
        int is_last_epoch = (epoch == n_epoch - 1) ? 1 : 0;
        launch_ours_gpu_kernel(
            d_edge, d_Y,
            N, n_threads, M, neg_num, step, seed
            ,
            d_sampled_node, is_last_epoch
        );
#else
        launch_ours_gpu_kernel(
            d_edge, d_Y,
            N, n_threads, M, neg_num, step, seed
        );
#endif
        cudaDeviceSynchronize();

        step += (stepMin - step) * stepSize;
        if (epoch % 50 == 0)
            printf("GPU epoch: %d/%d  step=%.6f\n", epoch, n_epoch, step);
    }
    t_layout.stop();

    // ================================================================
    // Step 8: Copy positions back to CPU
    // ================================================================
    Timer t_copyback("device -> host copy");
    cudaMemcpy(Y_float.data(), d_Y, N * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N * 2; ++i)
        Y[i] = static_cast<double>(Y_float[i]);
    t_copyback.stop();

#ifdef VERBOSE
    if (!config.fp_loss.empty() && d_sampled_node != nullptr)
    {
        // Copy sampled_node from GPU back to CPU
        std::vector<int> sampled_node(config.neg * M);
        cudaMemcpy(sampled_node.data(), d_sampled_node, config.neg * M * sizeof(int),
                   cudaMemcpyDeviceToHost);

        // Calculate loss based on final positions and recorded negative samples
        double cur_actual_loss_attr = 0., cur_actual_loss_rep = 0.;

        // Calculate attraction loss from edges
        for (size_t i = 0; i < M; ++i)
        {
            int u = edge_data[i * 2];
            int v = edge_data[i * 2 + 1];
            double dx = Y[v * 2] - Y[u * 2];
            double dy = Y[v * 2 + 1] - Y[u * 2 + 1];
            double dis2 = dx * dx + dy * dy;
            cur_actual_loss_attr += 0.05 * dis2 + 0.4 * std::log(1 + dis2);
        }

        // Calculate repulsion loss from recorded negative samples
        for (size_t i = 0; i < M; ++i)
        {
            int u = edge_data[i * 2];
            for (int j = 0; j < config.neg; ++j)
            {
                int other_id = sampled_node[i * config.neg + j];
                double dx = Y[other_id * 2] - Y[u * 2];
                double dy = Y[other_id * 2 + 1] - Y[u * 2 + 1];
                double dis2 = dx * dx + dy * dy;
                cur_actual_loss_rep += 0.5 / (1 + dis2);
            }
        }

        FILE* f = fopen(config.fp_loss.c_str(), "w");
        fprintf(f, "epoch,actual_loss_attr,actual_loss_rep,actual_loss_total\n");
        fprintf(f, "%d,%lf,%lf,%lf\n",
                n_epoch,
                cur_actual_loss_attr, cur_actual_loss_rep, cur_actual_loss_attr + cur_actual_loss_rep);
        fclose(f);
    }
#endif

    // Free GPU memory
    cudaFree(d_edge);
    cudaFree(d_Y);
    cudaFree(d_sampled_node);
#endif
}
