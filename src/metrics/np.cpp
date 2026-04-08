#include <vector>
#include <queue>
#include <unordered_set>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <omp.h>

#include <np.h>
using namespace std;

// =====================
// Global thread count
// =====================
int g_num_threads = 32; // 0 = use all available threads

struct KDTree
{
    int N;
    double* X;

    int* lc;
    int* rc;
    int* axis;
    int* id;

    int* ord;

    int root;

    // =====================
    // ThreadHeap for per-thread kNN
    // =====================
    struct ThreadHeap
    {
        int k;
        int sz;

        std::vector<int> id;
        std::vector<double> val;

        ThreadHeap(int max_k = 0)
        {
            if (max_k)
            {
                id.resize(max_k);
                val.resize(max_k);
            }
        }

        inline void ensure(int kk)
        {
            k = kk;
            sz = 0;
            if ((int)id.size() < kk)
            {
                id.resize(kk);
                val.resize(kk);
            }
        }
    };

    KDTree(int n, double* X) : N(n), X(X)
    {
        lc = new int[N];
        rc = new int[N];
        axis = new int[N];
        id = new int[N];
        ord = new int[N];

        for (int i = 0; i < N; ++i)
            ord[i] = i;

        root = build(0, N, 0);
    }

    static double* _X;
    static int _axis;

    static bool cmp(int a, int b)
    {
        return _X[a * 2 + _axis] < _X[b * 2 + _axis];
    }

    int build(int l, int r, int dep)
    {
        if (l >= r) return -1;

        int m = (l + r) >> 1;
        int ax = dep & 1;

        _X = X;
        _axis = ax;

        std::nth_element(ord + l, ord + m, ord + r, cmp);

        int u = ord[m];

        axis[u] = ax;

        lc[u] = build(l, m, dep + 1);
        rc[u] = build(m + 1, r, dep + 1);

        return u;
    }

    inline double dist2(int a, int b)
    {
        double dx = X[a * 2] - X[b * 2];
        double dy = X[a * 2 + 1] - X[b * 2 + 1];
        return dx * dx + dy * dy;
    }

    // ======================
    // kNN query (thread-safe)
    // ======================

    static inline void heap_down(ThreadHeap& H, int i)
    {
        while (1)
        {
            int l = i * 2 + 1;
            int r = l + 1;
            int m = i;

            if (l < H.k && H.val[l] > H.val[m]) m = l;
            if (r < H.k && H.val[r] > H.val[m]) m = r;

            if (m == i) break;

            std::swap(H.val[i], H.val[m]);
            std::swap(H.id[i], H.id[m]);
            i = m;
        }
    }

    static inline void push(ThreadHeap& H, int id0, double d)
    {
        if (H.sz < H.k)
        {
            H.id[H.sz] = id0;
            H.val[H.sz] = d;
            ++H.sz;

            if (H.sz == H.k)
                for (int i = H.k / 2 - 1; i >= 0; --i)
                    heap_down(H, i);

            return;
        }

        if (d >= H.val[0]) return;

        H.id[0] = id0;
        H.val[0] = d;
        heap_down(H, 0);
    }

    static inline double worst(const ThreadHeap& H)
    {
        if (H.sz < H.k) return 1e100;
        return H.val[0];
    }

    void query(int u, int q, ThreadHeap& H)
    {
        if (u == -1) return;

        double d = dist2(u, q);
        push(H, u, d);

        int ax = axis[u];

        double diff = X[q * 2 + ax] - X[u * 2 + ax];

        int near = diff < 0 ? lc[u] : rc[u];
        int far = diff < 0 ? rc[u] : lc[u];

        if (near != -1)
            query(near, q, H);

        if (far != -1 && diff * diff < worst(H))
            query(far, q, H);
    }

    void knn_threadsafe(int q, int k, int* out, ThreadHeap& H)
    {
        H.ensure(k);
        query(root, q, H);

        for (int i = 0; i < H.sz; ++i)
            out[i] = H.id[i];
    }
};

double* KDTree::_X;
int KDTree::_axis;

struct RRingBfs
{
    int N;
    const std::vector<int>& csr_offset;
    const std::vector<int>& csr_index;

    int ts{0};
    std::vector<int> vis;
    std::vector<int> dis;

    int k{0};
    std::vector<int> knn;

    std::queue<int> q;

    RRingBfs(const std::vector<int>& csr_offset, const std::vector<int>& csr_index)
        : N(csr_offset.size() - 1), csr_offset(csr_offset), csr_index(csr_index),
          vis(N), dis(N), knn(N)
    {
    }

    void ring1(const int start)
    {
        vis[start] = ++ts;
        k = 0;
        for (int i = csr_offset[start]; i < csr_offset[start + 1]; ++i)
        {
            const int& u = csr_index[i];
            if (vis[u] != ts)
                knn[k++] = u;
            vis[u] = ts;
        }
    }

    void ring2(const int start)
    {
        vis[start] = ++ts;
        k = 0;
        for (int i = csr_offset[start]; i < csr_offset[start + 1]; ++i)
        {
            const int& u = csr_index[i];
            if (vis[u] != ts)
                knn[k++] = u;
            vis[u] = ts;
            for (int j = csr_offset[u]; j < csr_offset[u + 1]; ++j)
            {
                const int& v = csr_index[j];
                if (vis[v] != ts)
                    knn[k++] = v;
                vis[v] = ts;
            }
        }
    }

    void r_ring(const int start, const int r)
    {
        ++ts;
        k = 0;
        q.push(start);
        dis[start] = 0;
        vis[start] = ts;
        while (!q.empty())
        {
            const int u = q.front();
            q.pop();
            if (u != start)
                knn[k++] = u;
            if (dis[u] == r)
                continue;
            for (int i = csr_offset[u]; i < csr_offset[u + 1]; ++i)
            {
                int v = csr_index[i];
                if (vis[v] == ts)
                    continue;
                vis[v] = ts;
                q.push(v);
                dis[v] = dis[u] + 1;
            }
        }
    }
};

CalcNp::CalcNp(const std::vector<Edge>& edges, const std::vector<Position>& pos)
    : N(pos.size()), M(edges.size()), X(N)
{
    std::vector<int> degree(N, 0);
    for (const auto& [u, v] : edges)
    {
        degree[u]++;
        degree[v]++;
    }

    // 构建原始图的 CSR offset
    csr_offset.resize(N + 1);
    csr_offset[0] = 0;
    for (int i = 0; i < N; ++i)
        csr_offset[i + 1] = csr_offset[i] + degree[i];

    // 分配邻接表空间
    csr_index.resize(csr_offset[N]);

    // 填充原始图的邻接表
    std::vector<int> pos_arr = csr_offset;

    for (const auto& [u, v] : edges)
    {
        csr_index[pos_arr[u]++] = v;
        csr_index[pos_arr[v]++] = u;
    }
    for (int i = 0; i < N; ++i)
    {
        X[i][0] = std::isnan(pos[i].first) ? 0.0 : pos[i].first;
        X[i][1] = std::isnan(pos[i].second) ? 0.0 : pos[i].second;
    }
}

double CalcNp::get(int ring_r)
{
    KDTree tree(N, X.pts);
    cerr << "built kdtree\n";

    double sum_np = 0;
    double sum_k = 0;

    if (g_num_threads > 0)
        omp_set_num_threads(g_num_threads);

#pragma omp parallel
    {
        RRingBfs bfs_local(csr_offset, csr_index);
        KDTree::ThreadHeap heap;
        vector<int> vis_local(N, -1);
        vector<int> ret_local(N);

        double local_np = 0;
        double local_k = 0;

#pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < N; ++i)
        {
            // 1. r-ring neighbors in graph
            if (ring_r == 1)
                bfs_local.ring1(i);
            else if (ring_r == 2)
                bfs_local.ring2(i);
            else
                bfs_local.r_ring(i, ring_r);
            const int k = bfs_local.k;
            local_k += k;

            // 2. k nearest neighbors in layout
            if (k == 0) continue;
            tree.knn_threadsafe(i, k, ret_local.data(), heap);

            // 3. Compute Jaccard
            for (int j = 0; j < k; ++j)
                vis_local[ret_local[j]] = i;
            int cnt_int = 0;
            for (int j = 0; j < k; ++j)
            {
                if (vis_local[bfs_local.knn[j]] == i)
                    ++cnt_int;
            }
            const int cnt_uni = 2 * k - cnt_int;
            local_np += (double)cnt_int / cnt_uni;
        }

#pragma omp atomic
        sum_np += local_np;

#pragma omp atomic
        sum_k += local_k;
    }

    cerr << "avg.k = " << (double)sum_k / N << " total.k = " << sum_k << endl;
    return sum_np / N;
}
