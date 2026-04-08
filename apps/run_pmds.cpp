#include <ogdf/basic/Graph.h>
#include <ogdf/basic/GraphAttributes.h>
#include <ogdf/energybased/PivotMDS.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>

using namespace std;
using namespace ogdf;

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        cerr << "Usage: " << argv[0] << " <graph.txt> <result.txt>" << endl;
        return 1;
    }

    const string graphFile = argv[1];
    const string resultFile = argv[2];

    // ============ 读取图 ============
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

    // ============ PMDS ============
    GraphAttributes GA(G, GraphAttributes::nodeGraphics);

    PivotMDS layout;

    // ⭐⭐⭐ 强烈建议你调这个
    layout.setNumberOfPivots(128);

    // ⭐ 2D layout（默认已是2D）
    layout.setForcing2DLayout(true);

    auto start = std::chrono::high_resolution_clock::now();
    layout.call(GA);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cout << "PMDS time: " << duration.count() / 1000.0 << " s" << endl;

    // ============ 归一化 ============
    double minx = 1e18, maxx = -1e18, miny = 1e18, maxy = -1e18;

    for (int i = 0; i < N; ++i)
    {
        double x = GA.x(nodeList[i]);
        double y = GA.y(nodeList[i]);

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