#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

// ============================================================
// Thread-based kernel: each thread processes one thread's edges
// Implements hogwild-style double-ended updates (no locks, accept conflicts)
// Edge format: d_edge[i*2] = u, d_edge[i*2+1] = v
// ============================================================
__global__
void ours_gpu_kernel(
    const int* __restrict__ d_edge,
    float* __restrict__ Y,
    const uint64_t N,
    const uint32_t n_threads,
    const uint64_t M,
    const int neg_num,
    const float step,
    const unsigned int seed
#ifdef VERBOSE
    ,
    int *__restrict__ sampled_node, // VERBOSE mode: record negative samples (only for last epoch)
    int is_last_epoch
#endif
)
{
    const uint64_t t_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_id >= n_threads) return;

    // Calculate edge range for this thread (uniform distribution)
    const uint64_t edges_per_thread = M / n_threads;
    const uint64_t remaining_edges = M % n_threads;

    const uint64_t edges_begin = t_id * edges_per_thread + min(t_id, remaining_edges);
    const uint64_t edges_end = edges_begin + edges_per_thread + (t_id < remaining_edges ? 1 : 0);

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, t_id, 0, &rng);

    // Process all edges assigned to this thread
    for (uint64_t i = edges_begin; i < edges_end; ++i)
    {
        const int u = __ldg(&d_edge[i * 2]);
        const int v = __ldg(&d_edge[i * 2 + 1]);

        // Cache u's position in registers to reduce global memory accesses and hogwild races
        float ux = Y[2 * u];
        float uy = Y[2 * u + 1];

        float dx = Y[2 * v] - ux;
        float dy = Y[2 * v + 1] - uy;
        float l = step * (0.1f + 0.8f / (1.f + dx * dx + dy * dy));
        float mvx = l * dx;
        float mvy = l * dy;

        ux += mvx;
        uy += mvy;
        Y[2 * v] -= mvx;
        Y[2 * v + 1] -= mvy;

        // Negative sampling
        for (int j = 0; j < neg_num; ++j)
        {
            int other = min((uint64_t)(curand_uniform(&rng) * (N - 1)), N - 2);
            if (other >= u) other++;

#ifdef VERBOSE
            // Record the negative sample node for loss calculation (only for last epoch)
            if (is_last_epoch&& sampled_node!= nullptr)
            {
                sampled_node[i * neg_num + j] = other;
            }
#endif

            dx = Y[2 * other] - ux;
            dy = Y[2 * other + 1] - uy;
            const float denom = 1.f + dx * dx + dy * dy;
            l = step * (-1.f / (denom * denom));
            mvx = l * dx;
            mvy = l * dy;

            ux += mvx;
            uy += mvy;
            Y[2 * other] -= mvx;
            Y[2 * other + 1] -= mvy;
        }

        // Write back u's accumulated position once
        Y[2 * u] = ux;
        Y[2 * u + 1] = uy;
    }
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
)
{
    uint32_t block_size = 256; // Maximum block size
    uint32_t grid_size = (n_threads + block_size - 1) / block_size;
#ifdef VERBOSE
    ours_gpu_kernel<<<grid_size, block_size>>>(
        d_edge, d_Y, N, n_threads, M, neg_num, step, seed, d_sampled_node, is_last_epoch);
#else
    ours_gpu_kernel<<<grid_size, block_size>>>(d_edge, d_Y, N, n_threads, M, neg_num, step, seed);
#endif
}
