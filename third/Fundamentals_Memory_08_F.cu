```cpp
/*
Aim of the program: Wrap the event timer logic in a C++ class GpuTimer for convenience.

Thinking process:
1. The user wants a CUDA program that demonstrates how to wrap the event timer logic
   in a C++ class called GpuTimer. The class should provide a convenient interface
   to start, stop, and query elapsed time for GPU kernels or arbitrary CUDA code.
2. In CUDA, timing is typically done using cudaEvent_t objects: create two events
   (start and stop), record the start event, launch the kernel, record the stop
   event, then synchronize and compute elapsed time with cudaEventElapsedTime.
3. The class should encapsulate this logic:
   - Constructor: create two events.
   - Destructor: destroy the events.
   - start(): record the start event.
   - stop(): record the stop event and synchronize.
   - elapsedTime(): compute and return the elapsed time in milliseconds.
4. To demonstrate usage, we’ll include a simple CUDA kernel that performs
   vector addition. The main function will allocate memory, copy data,
   invoke the kernel, and use GpuTimer to measure the kernel execution time.
5. The code will be written in a single .cu file, include necessary headers,
   error checking macros, and be fully compilable.
6. No additional output or explanation will be printed; only the .cu file content
   will be provided, as requested.

The program below implements these ideas.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = (call);                                  \
        if (err != cudaSuccess) {                                 \
            std::cerr << "CUDA error in " << __FILE__ << ":"      \
                      << __LINE__ << " (" << #call << "): "      \
                      << cudaGetErrorString(err) << std::endl;    \
            std::exit(EXIT_FAILURE);                              \
        }                                                          \
    } while (0)

// GpuTimer class encapsulates CUDA event timing
class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&startEvent_));
        CUDA_CHECK(cudaEventCreate(&stopEvent_));
    }

    ~GpuTimer() {
        CUDA_CHECK(cudaEventDestroy(startEvent_));
        CUDA_CHECK(cudaEventDestroy(stopEvent_));
    }

    // Record the start event
    void start() {
        CUDA_CHECK(cudaEventRecord(startEvent_, 0));
    }

    // Record the stop event and synchronize
    void stop() {
        CUDA_CHECK(cudaEventRecord(stopEvent_, 0));
        CUDA_CHECK(cudaEventSynchronize(stopEvent_));
    }

    // Return elapsed time in milliseconds
    float elapsedTime() const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, startEvent_, stopEvent_));
        return ms;
    }

private:
    cudaEvent_t startEvent_;
    cudaEvent_t stopEvent_;
};

// Simple CUDA kernel for vector addition
__global__
void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 2.0f);
    std::vector<float> h_C(N, 0.0f);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

    // Define execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    GpuTimer timer;

    // Start timing
    timer.start();

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Stop timing
    timer.stop();

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));

    // Verify result
    for (int i = 0; i < N; ++i) {
        assert(h_C[i] == h_A[i] + h_B[i]);
    }

    std::cout << "Kernel execution time: " << timer.elapsedTime() << " ms" << std::endl;

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```