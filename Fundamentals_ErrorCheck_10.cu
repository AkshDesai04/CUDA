```cuda
/*
Create a GpuTimer class in C++. The constructor should call cudaEventCreate for start and stop events. A start() method calls cudaEventRecord. A stop() method calls cudaEventRecord and cudaEventSynchronize. An elapsed_ms() method calls cudaEventElapsedTime. Add error checking to all CUDA calls.

Thinking:
- The timer needs to create two CUDA events: one for start and one for stop.
- All CUDA API calls should be wrapped with error checking so that any failure aborts the program with a clear message.
- A convenient macro `CHECK_CUDA` will execute a CUDA call, check its return value, and print an error if needed.
- The constructor will create both events, the destructor will destroy them.
- `start()` records the start event on the default stream (stream 0).
- `stop()` records the stop event and then synchronizes on that event to ensure all work preceding the stop has finished.
- `elapsed_ms()` queries the elapsed time between the start and stop events and returns it as a float (milliseconds).
- Disable copy and assignment to avoid accidental copies of the event handles.
- Provide a simple `main` that demonstrates the timer by launching a trivial kernel and measuring its execution time.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// Macro for checking CUDA errors
#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error at " << __FILE__ << ":"            \
                      << __LINE__ << " - "                              \
                      << cudaGetErrorString(err) << std::endl;          \
            std::exit(EXIT_FAILURE);                                    \
        }                                                               \
    } while (0)

// Simple kernel for demonstration (does nothing useful)
__global__ void dummyKernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2; // arbitrary operation
    }
}

class GpuTimer {
private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
public:
    GpuTimer() {
        CHECK_CUDA(cudaEventCreate(&start_));
        CHECK_CUDA(cudaEventCreate(&stop_));
    }

    ~GpuTimer() {
        CHECK_CUDA(cudaEventDestroy(start_));
        CHECK_CUDA(cudaEventDestroy(stop_));
    }

    void start() {
        CHECK_CUDA(cudaEventRecord(start_, 0));
    }

    void stop() {
        CHECK_CUDA(cudaEventRecord(stop_, 0));
        CHECK_CUDA(cudaEventSynchronize(stop_));
    }

    float elapsed_ms() const {
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

    // Disable copy construction and assignment
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;
};

int main() {
    const int N = 1 << 20; // 1M elements
    int *d_data = nullptr;
    size_t size = N * sizeof(int);

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_data, size));
    CHECK_CUDA(cudaMemset(d_data, 1, size));

    GpuTimer timer;
    timer.start();

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dummyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);

    timer.stop();
    float elapsed = timer.elapsed_ms();

    std::cout << "Kernel execution time: " << elapsed << " ms" << std::endl;

    // Clean up
    CHECK_CUDA(cudaFree(d_data));
    return 0;
}
```