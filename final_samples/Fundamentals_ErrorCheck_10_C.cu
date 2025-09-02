/*
Add a `stream` member and have the `start()` and `stop()` methods record events on that specific stream.

Thinking:
In order to satisfy the requirement, the timer class must maintain its own CUDA stream (`cudaStream_t stream`).
The constructor creates this stream with `cudaStreamCreate`. The `start()` and `stop()` methods record
CUDA events (`cudaEventRecord`) on that stream, ensuring that timing is specific to the operations
executed on this stream.  The elapsed time is obtained by synchronizing the stop event and then
computing the difference between the two events.  A getter (`getStream()`) exposes the stream so
that kernels can be launched onto it.  The rest of the program demonstrates a simple kernel
executed on this stream and reports the elapsed time.

The code includes basic error checking via the `CHECK_CUDA` macro.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

class GPUTimer {
public:
    GPUTimer() {
        CHECK_CUDA(cudaStreamCreate(&stream));
        CHECK_CUDA(cudaEventCreate(&startEvent));
        CHECK_CUDA(cudaEventCreate(&stopEvent));
    }

    ~GPUTimer() {
        CHECK_CUDA(cudaEventDestroy(startEvent));
        CHECK_CUDA(cudaEventDestroy(stopEvent));
        CHECK_CUDA(cudaStreamDestroy(stream));
    }

    void start() {
        CHECK_CUDA(cudaEventRecord(startEvent, stream));
    }

    void stop() {
        CHECK_CUDA(cudaEventRecord(stopEvent, stream));
    }

    float elapsedTime() const {
        CHECK_CUDA(cudaEventSynchronize(stopEvent));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, startEvent, stopEvent));
        return ms;
    }

    cudaStream_t getStream() const { return stream; }

private:
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    cudaStream_t stream;
};

__global__ void dummyKernel(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2;
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    int *d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(int)));

    GPUTimer timer;

    timer.start();
    dummyKernel<<<(N + 255) / 256, 256, 0, timer.getStream()>>>(d_data, N);
    timer.stop();

    float ms = timer.elapsedTime();
    std::cout << "Kernel time on stream: " << ms << " ms" << std::endl;

    CHECK_CUDA(cudaFree(d_data));
    return 0;
}
