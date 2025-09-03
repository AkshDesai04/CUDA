```cpp
/*
Perform the comparison on a machine with a high-end CPU and a low-end GPU, and vice-versa. Observe how the results change.

Thinking process:
1. The goal is to provide a simple benchmark that can be run on two different hardware setups:
   - One with a powerful CPU but a modest GPU.
   - Another with a less powerful CPU but a powerful GPU.
   The benchmark will compute the same workload on both the CPU and the GPU, then print out the execution times so that the user can observe the performance differences.

2. Chosen workload: Matrix multiplication (C = A × B).  
   - It is compute-bound and uses similar operations on both CPU and GPU.  
   - It scales well with matrix size and allows easy measurement of throughput.

3. Implementation details:
   - Allocate square matrices of a size specified by the user or default to 1024×1024.  
   - CPU implementation: simple triple-nested loop.  
   - GPU implementation: naive CUDA kernel with one thread per output element.  
   - Timing: use `std::chrono` for CPU and CUDA events for GPU.  
   - Print both times and the ratio (CPU/GPU).  

4. Error handling: 
   - Check CUDA API return codes.  
   - Ensure memory allocations succeed.

5. Build:
   - Compile with `nvcc -O3 -o matrix_bench matrix_bench.cu`.  
   - Run with `./matrix_bench [size]`.  

6. Expected usage:
   - Run on machine A (high-end CPU, low-end GPU) → observe GPU time maybe significantly slower than CPU.  
   - Run on machine B (low-end CPU, high-end GPU) → observe GPU time faster or comparable to CPU.

*/

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Naive matrix multiplication kernel: C = A * B
__global__ void matmul_kernel(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index of C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // column index of C

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CPU matrix multiplication
void matmul_cpu(const float *A, const float *B, float *C, int N) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int N = 1024; // default matrix size
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "Invalid matrix size: %s\n", argv[1]);
            return EXIT_FAILURE;
        }
    }
    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_cpu = (float *)malloc(bytes);
    float *h_C_gpu = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        fprintf(stderr, "Failed to allocate host matrices.\n");
        return EXIT_FAILURE;
    }

    // Initialize matrices A and B with some values
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(i % 100) / 100.0f;
        h_B[i] = static_cast<float>((i + 1) % 100) / 100.0f;
    }

    // ----------------- CPU computation -----------------
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matmul_cpu(h_A, h_B, h_C_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;

    // ----------------- GPU computation -----------------
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    // Copy A and B to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Timing with CUDA events
    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));

    CUDA_CHECK(cudaEventRecord(start_gpu, 0));

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);
    matmul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

    CUDA_CHECK(cudaEventRecord(stop_gpu, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_gpu));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start_gpu, stop_gpu));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));

    // Clean up GPU resources
    CUDA_CHECK(cudaEventDestroy(start_gpu));
    CUDA_CHECK(cudaEventDestroy(stop_gpu));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // ----------------- Verification -----------------
    // Verify that CPU and GPU results are close
    double max_diff = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double diff = fabs(h_C_cpu[i] - h_C_gpu[i]);
        if (diff > max_diff) max_diff = diff;
    }
    if (max_diff > 1e-3) {
        fprintf(stderr, "Result verification failed! Max diff = %e\n", max_diff);
    } else {
        printf("Result verification passed. Max diff = %e\n", max_diff);
    }

    // ----------------- Output -----------------
    printf("Matrix size: %d x %d\n", N, N);
    printf("CPU time:    %.3f ms\n", cpu_time.count() * 1000.0);
    printf("GPU time:    %.3f ms\n", gpu_ms);
    printf("Speedup (CPU/GPU): %.2fx\n", (cpu_time.count() * 1000.0) / gpu_ms);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return EXIT_SUCCESS;
}
```