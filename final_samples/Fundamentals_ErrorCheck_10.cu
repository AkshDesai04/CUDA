```cpp
/*
Create a `GpuTimer` class in C++. The constructor should call `cudaEventCreate` for start and stop events. A `start()` method calls `cudaEventRecord`. A `stop()` method calls `cudaEventRecord` and `cudaEventSynchronize`. An `elapsed_ms()` method calls `cudaEventElapsedTime`. Add error checking to all CUDA calls.

Thinking:
- The class will hold two `cudaEvent_t` members: `start_` and `stop_`.
- The constructor will create these events via `cudaEventCreate`, checking errors after each call.
- The destructor will destroy the events via `cudaEventDestroy`.
- `start()` will record the start event on the default stream.
- `stop()` will record the stop event and then synchronize the event to ensure completion.
- `elapsed_ms()` will compute the elapsed time between `start_` and `stop_` using `cudaEventElapsedTime`.
- A helper inline function `checkCuda` will wrap CUDA calls, printing the error string and aborting if the call fails.
- We'll include a small example kernel and a `main` function to demonstrate the timer usage.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// Helper to check CUDA errors
inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

class GpuTimer {
public:
    GpuTimer() {
        checkCuda(cudaEventCreate(&start_), "cudaEventCreate start");
        checkCuda(cudaEventCreate(&stop_), "cudaEventCreate stop");
    }

    ~GpuTimer() {
        checkCuda(cudaEventDestroy(start_), "cudaEventDestroy start");
        checkCuda(cudaEventDestroy(stop_), "cudaEventDestroy stop");
    }

    void start() {
        checkCuda(cudaEventRecord(start_, 0), "cudaEventRecord start");
    }

    void stop() {
        checkCuda(cudaEventRecord(stop_, 0), "cudaEventRecord stop");
        checkCuda(cudaEventSynchronize(stop_), "cudaEventSynchronize stop");
    }

    float elapsed_ms() const {
        float ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&ms, start_, stop_), "cudaEventElapsedTime");
        return ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

// Example kernel
__global__ void dummyKernel(int *d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = idx * 2;
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    int *d_data = nullptr;
    size_t size = N * sizeof(int);

    // Allocate device memory
    checkCuda(cudaMalloc(&d_data, size), "cudaMalloc d_data");

    // Launch kernel
    GpuTimer timer;
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    timer.start();
    dummyKernel<<<gridSize, blockSize>>>(d_data, N);
    timer.stop();

    // Check for kernel launch errors
    checkCuda(cudaGetLastError(), "Kernel launch");

    std::cout << "Kernel execution time: " << timer.elapsed_ms() << " ms" << std::endl;

    // Clean up
    checkCuda(cudaFree(d_data), "cudaFree d_data");

    return 0;
}
```