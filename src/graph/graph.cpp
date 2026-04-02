#include "graph.h"

#include <functional>
#include <fstream>
#include <sstream>

#include "pcg_random.hpp"

using namespace std;

void Graph::readGraph(string filname)
{
    FILE* file = fopen(filname.c_str(), "r");
    if (!file)
    {
        cerr << "[Error]: Cannot open file: " << filname << endl;
        return;
    }

    // 读取第一行，只取前两个整数
    fscanf(file, "%llu %llu%*[^\n]", &N, &M);

    edges.reserve(M * 2); // 预分配双倍空间（无向图）
    degree.resize(N, 0);

    int src, tgt;
    for (size_t i = 0; i < M; ++i)
    {
        // 读取每行的前两个整数，忽略后续内容
        if (fscanf(file, "%d %d%*[^\n]", &src, &tgt) == 2)
        {
            edges.emplace_back(src, tgt);
            edges.emplace_back(tgt, src); //无向无权图 反向插入一条边
            degree[src]++;
            degree[tgt]++;
        }
        else
        {
            cerr << "[Error]: Failed to read edge " << i << endl;
        }
    }
    fclose(file);

    M = edges.size();
    cout << "N: " << N << endl;
    cout << "E: " << M / 2 << endl;
}

void Graph::init_PMDS(string filename)
{
    delete[] Y;
    Y = new double[N * 2];
    string line;
    // 读取pmds - 提前计算好
    ifstream pmdsfile(filename);
    int index = 0;
    while (getline(pmdsfile, line))
    {
        istringstream iss1(line);
        double x, y;
        if (iss1 >> x >> y)
        {
            Y[index * 2] = x; // 第一列
            Y[index * 2 + 1] = y; // 第二列
            index++;
        }
    }
}

void Graph::init_random(const int random_seed)
{
    delete[] Y;
    Y = new double[N * 2];

    std::mt19937 rng(random_seed);

    const double sigma = std::sqrt(static_cast<double>(N));

    std::normal_distribution dist(0.0, sigma);

    for (int i = 0; i < N; ++i)
    {
        Y[2 * i] = dist(rng);
        Y[2 * i + 1] = dist(rng);
    }
}

void Graph::init_spiral()
{
#undef M_PI
    static const double M_PI = 3.14159265358979323846;
    delete[] Y;
    Y = new double[N * 2];

    constexpr double max_radius = 10.0;
    const double max_sqrt_n = sqrt(N - 1); // 计算最大 sqrt(i)
    const double scale_factor = max_radius / max_sqrt_n; // 归一化因子

    for (int i = 0; i < N; i++)
    {
        const double radius = sqrt(i) * scale_factor; // 归一化半径
        const double angle = i * M_PI * (3.0 - sqrt(5.0));

        Y[2 * i] = radius * cos(angle);
        Y[2 * i + 1] = radius * sin(angle);
    }
}


void Graph::normalizePos() const
{
    double maxx = -1e9, minx = 1e9, maxy = -1e9, miny = 1e9;

    for (int i = 0; i < N; i++)
    {
        if (Y[2 * i] > maxx)
            maxx = Y[2 * i];
        if (Y[2 * i] < minx)
            minx = Y[2 * i];
        if (Y[2 * i + 1] > maxy)
            maxy = Y[2 * i + 1];
        if (Y[2 * i + 1] < miny)
            miny = Y[2 * i + 1];
    }
    double length = max(maxx - minx, maxy - miny);
    for (int i = 0; i < N; i++)
    {
        Y[2 * i] -= minx;
        Y[2 * i] /= length;

        Y[2 * i + 1] -= miny;
        Y[2 * i + 1] /= length;
    }
}

void Graph::savePos(string filename)
{
    normalizePos();
    FILE* file = fopen(filename.c_str(), "w");
    if (!file)
    {
        cerr << "[Error]: Cannot open file for writing: " << filename << endl;
        return;
    }

    for (int i = 0; i < N; ++i)
        fprintf(file, "%.17g %.17g\n", Y[i * 2], Y[i * 2 + 1]);

    fclose(file);
}

void Graph::drawSvg(string attrFilename, string svgFilename)
{
    vector<int> labels(N, 0);
    vector<string> colors_tab20 =
    {
        "1F77B4", "#AEC7E8", // 蓝色
        "#FF7F0E", "#FFBB78", // 橙色
        "#2CA02C", "#98DF8A", // 绿色
        "#D62728", "#FF9896", // 红色
        "#9467BD", "#C5B0D5", // 紫色
        "#8C564B", "#C49C94", // 棕色
        "#E377C2", "#F7B6D2", // 粉色
        "#7F7F7F", "#C7C7C7", // 灰色
        "#BCBD22", "#DBDB8D", // 橄榄绿
        "#17BECF", "#9EDAE5" // 青色
    };
    // 读取标签文件
    ifstream labelFile(attrFilename);
    string line;
    if (labelFile)
    {
        labels.clear();

        while (getline(labelFile, line))
        {
            if (!line.empty())
            {
                labels.push_back(stoi(line));
            }
        }
    }

    double maxx = -1e9, minx = 1e9, maxy = -1e9, miny = 1e9;
    for (int i = 0; i < N; i++)
    {
        if (Y[2 * i] > maxx)
            maxx = Y[2 * i];
        if (Y[2 * i] < minx)
            minx = Y[2 * i];
        if (Y[2 * i + 1] > maxy)
            maxy = Y[2 * i + 1];
        if (Y[2 * i + 1] < miny)
            miny = Y[2 * i + 1];
    }
    double length = max(maxx - minx, maxy - miny);
    for (int i = 0; i < N; i++)
    {
        Y[2 * i] -= minx;
        Y[2 * i + 1] -= miny;
    }
    for (int i = 0; i < N; i++)
    {
        Y[2 * i] /= length;
        Y[2 * i] *= 900;
        Y[2 * i] += 50;
        Y[2 * i + 1] /= length;
        Y[2 * i + 1] *= 900;
        Y[2 * i + 1] += 50;
    }

    ofstream f;
    f.open(svgFilename, ios::out);
    f << "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" \n"
        "\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n"
        "\n"
        "<svg width=\"1000\" height=\"1000\" version=\"1.1\"\n"
        "xmlns=\"http://www.w3.org/2000/svg\">"
        << endl;

    // 边太多了 先不画
    if (edges.size() <= 500000)
    {
        for (auto& [fst, snd] : edges)
        {
            f << "<line x1=\"" << Y[2 * fst] << "\" y1=\"" << Y[2 * fst + 1] << "\" x2=\"" << Y[2
                    *
                    snd] << "\" y2=\"" << Y[2 * snd + 1] << "\"\n"
                << "style=\"stroke:#808080;stroke-width:0.5;stroke-opacity:0.7\"/>"
                << endl;
        }
    }

    for (int i = 0; i < N; i++)
    {
        f << "<circle id=\"" << i
            << "\" cx=\"" << Y[2 * i]
            << "\" cy=\"" << Y[2 * i + 1]
            << R"(" r="2" fill=")" << colors_tab20[labels[i] % 20]
            << "\"/>\n";
    }

    f << "</svg>" << endl;
    f.close();
}

std::pair<double, double> Graph::calc_loss(const int k)
{
    double loss_attr = 0., loss_rep = 0.;
    for (const auto& [u,v] : edges)
    {
        const double dx = Y[u * 2] - Y[v * 2];
        const double dy = Y[u * 2 + 1] - Y[v * 2 + 1];
        const double dis2 = dx * dx + dy * dy;
        loss_attr += 0.05 * dis2 + 0.4 * std::log(1 + dis2);
    }
    for (int u = 0; u < N; ++u)
    {
        for (int v = 0; v < N; ++v)
        {
            if (u != v)
            {
                const double dx = Y[u * 2] - Y[v * 2];
                const double dy = Y[u * 2 + 1] - Y[v * 2 + 1];
                const double dis2 = dx * dx + dy * dy;
                loss_rep += 1. * k * (degree[u] + degree[v]) / (2 * N) * 0.5 / (1 + dis2);
            }
        }
    }
    return std::make_pair(loss_attr, loss_rep);
}
