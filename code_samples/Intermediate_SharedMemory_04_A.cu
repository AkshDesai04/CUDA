```cuda
/*
Aim of the program: Calculate and report the performance in GFLOPS (billions of floating-point operations per second). The number of FLOPS for matrix multiplication is `2 * N^3`.

Thinking:
1. The program should accept an integer N representing the matrix dimension.
   - If no argument is provided, we will default to N = 1024 for a realistic test.
2. Allocate three float arrays A, B, C of size N * N on both host and device.
3. Initialize A and B with some deterministic values (e.g., A[i] = 1.0f, B[i] = 2.0f).
4. Define a simple CUDA kernel that performs matrix multiplication C = A * B.
   - We'll use a naive global memory implementation for clarity.
5. Use cudaEvent_t to time the kernel execution precisely.
6. After kernel completion, copy result back to host (optional, but we will copy to ensure kernel finished).
7. Compute the number of floating-point operations: FLOPS = 2 * N^3.
8. Calculate the elapsed time in seconds and then GFLOPS = FLOPS / (time * 1e9).
9. Print N, elapsed time, and GFLOPS.
10. Clean up all allocated memory and CUDA resources.
11. Add basic error checking for CUDA API calls.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Naive matrix multiplication kernel
__global__ void matMulKernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char* argv[]) {
    // Parse matrix size N
    int N = 1024; // default size
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "Invalid matrix size. Must be positive integer.\n");
            return EXIT_FAILURE;
        }
    }

    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host matrices
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;      // example value
        h_B[i] = 2.0f;      // example value
        h_C[i] = 0.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch kernel
    matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Record stop
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float seconds = milliseconds / 1000.0f;

    // Copy result back to host (optional, but ensures completion)
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Compute GFLOPS
    double flops = 2.0 * N * N * N;   // 2 * N^3
    double gflops = flops / (seconds * 1e9);

    // Output results
    printf("Matrix size: %d x %d\n", N, N);
    printf("Execution time: %f seconds\n", seconds);
    printf("GFLOPS: %f\n", gflops);

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}
```