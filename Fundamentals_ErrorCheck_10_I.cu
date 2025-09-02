/*
Make the class non-copyable by deleting the copy constructor and copy assignment operator, as `cudaEvent_t` handles cannot be trivially copied.

I decided to implement a simple CUDA timing utility as a class called Timer. The Timer encapsulates two cudaEvent_t objects: one for the start time and one for the stop time. It provides methods to start timing, stop timing, and retrieve the elapsed time in milliseconds. Because cudaEvent_t handles represent GPU resources that cannot be copied trivially, I made the class non-copyable by explicitly deleting the copy constructor and copy assignment operator. This prevents accidental copying of Timer objects, which would otherwise lead to undefined behavior or double destruction of the CUDA events.

The program also includes a small vector addition kernel to demonstrate usage of the Timer class. In the main function, I allocate device memory, copy input data, launch the kernel, and then use the Timer to measure the execution time. I used a simple error-checking macro to simplify CUDA API error handling. After the kernel execution, I output the elapsed time in milliseconds.

The code is self-contained and ready to compile with nvcc. It showcases proper resource management for CUDA events and demonstrates how to enforce non-copyability in a C++ class that wraps CUDA resources. 
*/

#include <iostream>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error in file '" << __FILE__                \
                      << "' in line " << __LINE__ << ": "                  \
                      << cudaGetErrorString(err) << std::endl;             \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Simple kernel for vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Timer class that uses CUDA events
class GPUTimer {
private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
public:
    // Constructor: create events
    GPUTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    // Destructor: destroy events
    ~GPUTimer() {
        CUDA_CHECK(cudaEventDestroy(start_));
        CUDA_CHECK(cudaEventDestroy(stop_));
    }

    // Delete copy constructor and copy assignment operator
    GPUTimer(const GPUTimer&) = delete;
    GPUTimer& operator=(const GPUTimer&) = delete;

    // Start timing
    void start() {
        CUDA_CHECK(cudaEventRecord(start_, 0));
    }

    // Stop timing
    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_));
    }

    // Get elapsed time in milliseconds
    float elapsed_ms() const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

int main() {
    const int N = 1 << 20; // 1M elements
    const int size = N * sizeof(float);

    // Allocate host memory
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Create timer
    GPUTimer timer;

    // Launch kernel and time it
    timer.start();
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    timer.stop();

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify result (optional)
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5f) {
            ok = false;
            std::cerr << "Mismatch at index " << i << std::endl;
            break;
        }
    }

    std::cout << "Kernel execution time: " << timer.elapsed_ms() << " ms\n";
    std::cout << "Result verification: " << (ok ? "PASS" : "FAIL") << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
