```cpp
/*
Write a multi-threaded CPU version of vector addition (e.g., using OpenMP) and compare it to the GPU version.

Thought process:
1. The program should perform vector addition (C = A + B) on both CPU and GPU.
2. For the CPU side, use OpenMP to parallelize the loop across available cores.
3. For the GPU side, write a simple CUDA kernel that adds elements of two arrays.
4. Allocate large vectors (e.g., 16 million elements) to make the performance difference noticeable.
5. Measure execution time for both CPU and GPU:
   - Use std::chrono::high_resolution_clock for CPU timing.
   - Use cudaEvent for GPU timing to account for asynchronous kernel launch.
6. After execution, validate the results by computing the maximum absolute difference between CPU and GPU results.
7. Report timings and speedup factor.
8. Include error checking for CUDA API calls.
9. Ensure the code compiles with `nvcc -Xcompiler -fopenmp`.
10. Keep the code self-contained in a single .cu file, with all necessary headers and implementations.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1 << 24; // 16,777,216 elements (~64 MB per array)
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    float *h_C_gpu = (float*)malloc(bytes);

    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize input vectors with random data
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // CPU OpenMP vector addition
    auto cpu_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        h_C_cpu[i] = h_A[i] + h_B[i];
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    printf("CPU (OpenMP) time: %.6f seconds\n", cpu_duration.count());

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy input vectors to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    CUDA_CHECK(cudaEventRecord(start));

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));
    printf("GPU time: %.3f ms\n", gpu_time_ms);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify results
    float max_error = 0.0f;
    for (int i = 0; i < N; ++i) {
        float diff = fabs(h_C_cpu[i] - h_C_gpu[i]);
        if (diff > max_error) max_error = diff;
    }
    printf("Max absolute error between CPU and GPU results: %e\n", max_error);

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}
```