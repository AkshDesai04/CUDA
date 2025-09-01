/*
Aim of the program: Wrap the event timer logic in a C++ class `GpuTimer` for convenience.

Thinking:
- The goal is to encapsulate CUDA event timing into a reusable C++ class. 
- The class should create two CUDA events (start and stop) in the constructor.
- Provide methods: startTimer(), stopTimer(), elapsedTime() returning milliseconds, and optionally reset().
- Destructor should destroy the events.
- Include error checking macro to catch CUDA API errors.
- Demonstrate usage in a main() with a dummy kernel launch to showcase timing.
- Use CUDA runtime API; include <cuda_runtime.h> and <iostream>.
- Ensure code compiles as a .cu file with nvcc.
- Keep the code self-contained, no external dependencies.
*/

#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in file '" << __FILE__                   \
                      << "' in line " << __LINE__ << ": "                     \
                      << cudaGetErrorString(err) << std::endl;                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~GpuTimer() {
        CUDA_CHECK(cudaEventDestroy(start_));
        CUDA_CHECK(cudaEventDestroy(stop_));
    }

    void startTimer() {
        CUDA_CHECK(cudaEventRecord(start_, 0));
    }

    void stopTimer() {
        CUDA_CHECK(cudaEventRecord(stop_, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_));
    }

    float elapsedTime() const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

// Dummy kernel for demonstration
__global__ void dummyKernel(int *a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] = idx;
}

int main() {
    const int N = 1 << 20; // 1M elements
    int *devPtr = nullptr;
    size_t size = N * sizeof(int);

    CUDA_CHECK(cudaMalloc(&devPtr, size));

    GpuTimer timer;
    timer.startTimer();

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dummyKernel<<<blocksPerGrid, threadsPerBlock>>>(devPtr);

    timer.stopTimer();

    float elapsed = timer.elapsedTime();
    std::cout << "Kernel execution time: " << elapsed << " ms" << std::endl;

    CUDA_CHECK(cudaFree(devPtr));
    return 0;
}
