```cuda
/*
Aim of the program:
For the CPU timing, use `std::chrono::high_resolution_clock`.

Thinking:
I need to write a complete CUDA program in a .cu file that demonstrates how to use the
C++ high-resolution clock for measuring CPU-side performance.  The program will
perform a simple vector addition on the GPU: it will allocate two input arrays
and one output array on the host, copy the inputs to the device, launch a
kernel that adds the arrays element-wise, copy the result back, and then free
resources.  I will time the overall CPU work (allocation, host‑to‑device copy,
kernel launch & synchronization, device‑to‑host copy, and deallocation) using
`std::chrono::high_resolution_clock`.  The GPU timing will be optional and
shown for comparison using CUDA events, but the requirement specifically
calls out the CPU timing.  All code will be self‑contained and ready to be
compiled with `nvcc`.  I’ll also add a small helper macro for checking CUDA
errors to keep the code clean.  No extra text or comments beyond the required
multiline comment at the top will be output, as per the instruction.
*/

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            std::cerr << "CUDA error in " << __FILE__ << ":"        \
                      << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE);                                \
        }                                                           \
    } while (0)

// Kernel for element-wise vector addition: C = A + B
__global__ void vectorAdd(const float *A, const float *B, float *C, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const size_t N = 1 << 24;          // 16 million elements
    const size_t size_bytes = N * sizeof(float);

    // Timing start (CPU)
    auto cpu_start = std::chrono::high_resolution_clock::now();

    // Allocate host memory
    float *h_A = (float*)malloc(size_bytes);
    float *h_B = (float*)malloc(size_bytes);
    float *h_C = (float*)malloc(size_bytes);

    // Initialize host arrays
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_bytes));

    // Copy inputs from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_bytes, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Synchronize to ensure kernel completion
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_bytes, cudaMemcpyDeviceToHost));

    // Timing end (CPU)
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;

    // Optional: GPU timing using CUDA events for comparison
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent, 0));
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stopEvent, 0));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, startEvent, stopEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    // Verify result (optional)
    bool correct = true;
    for (size_t i = 0; i < N; ++i) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5f) {
            correct = false;
            std::cerr << "Mismatch at index " << i << ": "
                      << h_C[i] << " != " << (h_A[i] + h_B[i]) << std::endl;
            break;
        }
    }

    std::cout << "CPU time: " << cpu_duration.count() << " ms" << std::endl;
    std::cout << "GPU time (CUDA events): " << gpu_ms << " ms" << std::endl;
    std::cout << "Result verification: " << (correct ? "PASSED" : "FAILED") << std::endl;

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```