#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "graph.h"

using namespace std;

int main(int argc, char* argv[])
{
    if (argc != 5)
    {
        cerr << "Usage: " << argv[0]
             << " <input_layout.txt> <output.svg> <graph.txt|\"\"|-> <labels.attr|\"\"|->" << endl;
        return 1;
    }

    const string inputLayout = argv[1];
    const string outputSvg = argv[2];
    const string graphArg = argv[3];
    const string attrArg = argv[4];
    const string graphFile = (graphArg == "-" ? "" : graphArg);
    const string attrFile = (attrArg == "-" ? "" : attrArg);

    ifstream in(inputLayout);
    if (!in)
    {
        cerr << "[Error]: Cannot open file: " << inputLayout << endl;
        return 1;
    }

    Graph g;
    string line;
    double x, y;

    while (getline(in, line))
    {
        if (line.empty())
            continue;

        istringstream iss(line);
        if (!(iss >> x >> y))
        {
            cerr << "[Error]: Invalid layout line: " << line << endl;
            return 1;
        }

        g.N++;
    }

    if (g.N == 0)
    {
        cerr << "[Error]: Empty layout file: " << inputLayout << endl;
        return 1;
    }

    if (graphFile != "")
    {
        g.readGraph(graphFile);
    }

    g.Y = new double[g.N * 2];

    in.clear();
    in.seekg(0);

    size_t idx = 0;
    while (getline(in, line))
    {
        if (line.empty())
            continue;

        istringstream iss(line);
        if (!(iss >> x >> y))
        {
            cerr << "[Error]: Invalid layout line: " << line << endl;
            return 1;
        }

        g.Y[idx * 2] = x;
        g.Y[idx * 2 + 1] = y;
        idx++;
    }

    if (graphFile != "" && g.N != idx)
    {
        cerr << "[Error]: Graph node count does not match layout node count." << endl;
        return 1;
    }

    g.N = idx;
    g.drawSvg(attrFile, outputSvg);
    cout << "Results saved to: " << outputSvg << endl;
    return 0;
}
