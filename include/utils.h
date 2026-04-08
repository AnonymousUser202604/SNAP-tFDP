#pragma once
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#define TEE_DUP    _dup
#define TEE_DUP2   _dup2
#define TEE_CLOSE  _close
#define TEE_READ   _read
#define TEE_WRITE  _write
#define TEE_FILENO _fileno
#else
#include <unistd.h>
#define TEE_DUP    dup
#define TEE_DUP2   dup2
#define TEE_CLOSE  close
#define TEE_READ   read
#define TEE_WRITE  write
#define TEE_FILENO fileno
#endif

class Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start_;
    std::string label_;
public:
    explicit Timer(std::string label) : start_(Clock::now()), label_(std::move(label)) {}

    // 手动获取已过时间（秒），不打印
    double elapsed() const {
        return std::chrono::duration<double>(Clock::now() - start_).count();
    }

    // 手动重置
    void reset() { start_ = Clock::now(); }

    // 打印并返回耗时
    double stop() const {
        double s = elapsed();
        printf("[Timer] %s: %.3fs\n", label_.c_str(), s);
        return s;
    }
};

class TeeLogger {
    int saved_fd_;
    int pipe_read_fd_;
    FILE* log_file_;
    std::thread reader_;

public:
    explicit TeeLogger(const std::string& log_path)
        : saved_fd_(-1), pipe_read_fd_(-1), log_file_(nullptr)
    {
        log_file_ = fopen(log_path.c_str(), "w");
        if (!log_file_) {
            fprintf(stderr, "TeeLogger: cannot open %s\n", log_path.c_str());
            return;
        }

        saved_fd_ = TEE_DUP(TEE_FILENO(stdout));

        int fds[2];
#ifdef _WIN32
        if (_pipe(fds, 65536, _O_BINARY) != 0) {
#else
        if (pipe(fds) != 0) {
#endif
            fprintf(stderr, "TeeLogger: pipe failed\n");
            fclose(log_file_); log_file_ = nullptr;
            TEE_CLOSE(saved_fd_); saved_fd_ = -1;
            return;
        }

        pipe_read_fd_ = fds[0];
        TEE_DUP2(fds[1], TEE_FILENO(stdout));
        TEE_CLOSE(fds[1]);

        setvbuf(stdout, NULL, _IONBF, 0);

        int rfd = pipe_read_fd_;
        int cfd = saved_fd_;
        FILE* lf = log_file_;
        reader_ = std::thread([rfd, cfd, lf]() {
            char buf[4096];
            for (;;) {
                int n = TEE_READ(rfd, buf, sizeof(buf));
                if (n <= 0) break;
                TEE_WRITE(cfd, buf, n);
                fwrite(buf, 1, n, lf);
                fflush(lf);
            }
        });
    }

    ~TeeLogger() {
        if (saved_fd_ < 0) return;
        fflush(stdout);
        TEE_DUP2(saved_fd_, TEE_FILENO(stdout));
        if (reader_.joinable()) reader_.join();
        TEE_CLOSE(pipe_read_fd_);
        TEE_CLOSE(saved_fd_);
        fclose(log_file_);
    }

    TeeLogger(const TeeLogger&) = delete;
    TeeLogger& operator=(const TeeLogger&) = delete;
};
