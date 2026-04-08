#include <ogdf/basic/Graph.h>
#include <ogdf/basic/GraphAttributes.h>
#include <ogdf/energybased/SpringEmbedderFRExact.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>

using namespace std;
using namespace ogdf;

int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        cerr << "Usage: " << argv[0] << " <graph.txt> <pmds_init.txt> <result.txt>" << endl;
        return 1;
    }

    const string graphFile = argv[1];
    const string pmdsFile = argv[2];
    const string resultFile = argv[3];

    // ============ 读取图 ============
    // 第一行: N M，后续每行: u v [忽略后续列]
    FILE* f = fopen(graphFile.c_str(), "r");
    if (!f)
    {
        cerr << "[Error]: Cannot open file: " << graphFile << endl;
        return 1;
    }

    int N, M;
    fscanf(f, "%d %d%*[^\n]", &N, &M);

    Graph G;
    vector<node> nodeList(N);
    for (int i = 0; i < N; ++i)
        nodeList[i] = G.newNode();

    int src, tgt;
    for (int i = 0; i < M; ++i)
    {
        if (fscanf(f, "%d %d%*[^\n]", &src, &tgt) == 2)
            G.newEdge(nodeList[src], nodeList[tgt]);
        else
            cerr << "[Error]: Failed to read edge " << i << endl;
    }
    fclose(f);

    cout << "N: " << N << ", E: " << M << endl;

    // ============ 初始化节点位置 ============
    // 每行: x y
    GraphAttributes GA(G, GraphAttributes::nodeGraphics);

    ifstream posfile(pmdsFile);
    if (!posfile)
    {
        cerr << "[Error]: Cannot open file: " << pmdsFile << endl;
        return 1;
    }

    string line;
    int id = 0;
    while (getline(posfile, line))
    {
        istringstream iss(line);
        double x, y;
        if (!(iss >> x >> y)) continue;
        if (id >= N)
        {
            cerr << "[Error]: PMDS file has more lines than nodes." << endl;
            return 1;
        }
        GA.x(nodeList[id]) = x;
        GA.y(nodeList[id]) = y;
        id++;
    }
    posfile.close();

    // ============ 运行 FR 布局 ============
    auto start = std::chrono::high_resolution_clock::now();
    SpringEmbedderFRExact layout;
    layout.call(GA);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cout << "Layout time: " << duration.count() / 1000.0 << " s" << endl;

    // ============ 归一化并保存 ============
    // 归一化到 [0,1]，格式与 Graph::savePos 一致
    double minx = 1e18, maxx = -1e18, miny = 1e18, maxy = -1e18;
    for (int i = 0; i < N; ++i)
    {
        double x = GA.x(nodeList[i]), y = GA.y(nodeList[i]);
        if (x < minx) minx = x;
        if (x > maxx) maxx = x;
        if (y < miny) miny = y;
        if (y > maxy) maxy = y;
    }
    double len = max(maxx - minx, maxy - miny);

    FILE* out = fopen(resultFile.c_str(), "w");
    if (!out)
    {
        cerr << "[Error]: Cannot open file for writing: " << resultFile << endl;
        return 1;
    }

    for (int i = 0; i < N; ++i)
    {
        double x = (GA.x(nodeList[i]) - minx) / len;
        double y = (GA.y(nodeList[i]) - miny) / len;
        fprintf(out, "%.17g %.17g\n", x, y);
    }
    fclose(out);

    cout << "Results saved to: " << resultFile << endl;
    return 0;
}
