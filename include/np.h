#ifndef NSGL_NP_H
#define NSGL_NP_H

#include <vector>

// -----------------------------
// KD-Tree wrapper for nanoflann
// -----------------------------
struct PointCloud
{
    int n;
    double* pts;

    PointCloud(const int n)
        : n(n)
    {
        pts = new double[n * 2];
    }

    double* operator [](const int& idx) const
    {
        return pts + idx * 2;
    }

    inline size_t kdtree_get_point_count() const { return n; }

    inline double kdtree_distance(const double* p1, const size_t idx_p2, size_t /*size*/) const
    {
        const double dx = p1[0] - pts[idx_p2 * 2];
        const double dy = p1[1] - pts[idx_p2 * 2 + 1];
        return dx * dx + dy * dy;
    }

    inline double kdtree_get_pt(const size_t idx, int dim) const { return pts[idx * 2 + dim]; }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }
};

class CalcNp
{
public:
    typedef std::pair<double, double> Position;
    typedef std::pair<int, int> Edge;
    CalcNp(const std::vector<Edge>& edges, const std::vector<Position>& pos);
    double get(int ring_r = 2);

private:
    int N, M;
    PointCloud X;
    std::vector<int> csr_offset;
    std::vector<int> csr_index;
};

#endif //NSGL_NP_H
