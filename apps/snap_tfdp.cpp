#include <filesystem>
#include <iostream>
#include <string>

#include <CLI11.hpp>

#include "graph.h"
#include "layout.h"
#include "utils.h"

using namespace std;

int main(int argc, char* argv[])
{
    CLI::App app{"SNAP-tFDP"};

    // ============ 数据IO参数 ============
    string fp_data;
    app.add_option("dataset", fp_data, "Dataset path")->required();

    string fp_output;
    app.add_option("output", fp_output, "Output path")->required();

    string init = "pmds";
    app.add_option("--init", init, "Init method: pmds, random, spiral")
       ->check(CLI::IsMember({"pmds", "random", "spiral"}));

    string pmds_file = "";
    app.add_option("--pmds-file", pmds_file, "PMDS initialization file path (required when --init pmds)");

    int n_epoch = 50;
    app.add_option("-t,--n-epoch", n_epoch, "Number of epochs")
       ->check(CLI::NonNegativeNumber);

    int param_k = 3;
    app.add_option("-k,--k", param_k, "Negative sampling number (param k)")
       ->check(CLI::PositiveNumber);

    int seed = 42;
    app.add_option("--seed", seed, "Random seed");

    bool parallel = false;
    app.add_flag("-p,--parallel", parallel, "Enable CPU parallel mode");

    bool gpu = false;
    app.add_flag("-g,--gpu", gpu, "Enable GPU parallel mode");

    int n_threads = -1;
    app.add_option("--n-threads", n_threads,
                   "Number of threads (default: -1 for max, effective with --parallel or --gpu)");

#ifdef VERBOSE
    string fp_loss = "";
    app.add_option("--fp-loss", fp_loss, "Loss output file path (required when VERBOSE is enabled)");
#endif

    CLI11_PARSE(app, argc, argv);

    // ============ 构建文件路径 ============
    string graphFile = fp_data;
    string resultPos = fp_output;
#ifdef VERBOSE
    if (fp_loss == "")
    {
        printf("Error : Empty loss output file path\n");
        return 1;
    }
#endif

    // ============ 初始化TeeLogger ============
    std::string logFile = resultPos.substr(0, resultPos.rfind('.')) + ".log";
    TeeLogger tee(logFile);

    // ============ 打印参数信息 ============
    cout << "=== Parameters ===" << endl;
    cout << "Dataset: " << fp_data << endl;
    cout << "Result: " << fp_output << endl;
    cout << "Init: " << init << endl;
    cout << "Epochs: " << n_epoch << endl;
    cout << "Param k: " << param_k << endl;
    cout << "Seed: " << seed << endl;
    cout << "Parallel: " << (parallel ? "true" : "false") << endl;
    cout << "GPU: " << (gpu ? "true" : "false") << endl;
    cout << "Threads: " << n_threads << endl;
    cout << "==================" << endl << endl;

    // ============ 读取图数据 ============
    cout << "Processing: " << fp_data << endl;
    Graph g;
    g.readGraph(graphFile);
    if (init == "pmds")
    {
        if (pmds_file.empty())
        {
            cerr << "Error: --pmds-file is required when --init pmds" << endl;
            return 1;
        }
        g.init_PMDS(pmds_file);
    }
    else if (init == "spiral")
        g.init_spiral();
    else
        g.init_random(seed);

    // ============ 执行布局算法 ============
    auto start = chrono::high_resolution_clock::now();
    Layout layout(g);
    layout.run(LayoutConfig{
        .n_epoch = n_epoch,
        .k = param_k,
        .random_seed = seed,
        .parallel = parallel,
        .gpu = gpu,
        .n_threads = n_threads
    });
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end - start;
    cout << "Runtime: " << elapsed.count() / 1000 << " s\n";

    // ============ 保存结果 ============
    g.savePos(resultPos);

    cout << "Results saved to: " << resultPos << endl;

    return 0;
}
