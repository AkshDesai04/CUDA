```cuda
/*
Plot the GFLOPS of both kernels for a range of matrix sizes (e.g., 256, 512, 1024, 2048).

Thought Process:
- We need a CUDA program that runs two different kernels and measures their performance in terms of GFLOPS for several matrix sizes.
- Matrix multiplication is a common benchmark; we can implement two versions:
  1. A naive global‑memory kernel.
  2. An optimized tiled kernel using shared memory.
- Performance metric: GFLOPS = (2 * N^3) / (time_in_seconds * 1e9).
- Use CUDA events for accurate timing.
- Loop over matrix sizes 256, 512, 1024, 2048.
- Allocate device memory for each size, initialize matrices with random floats, run each kernel, measure time, compute GFLOPS, and print results.
- The program outputs a table that can be used to plot GFLOPS vs. matrix size externally (e.g., in Excel, Python, etc.).
- No external plotting is performed in the code; the output is console text.
- Ensure proper error checking, memory deallocation, and device synchronization.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Naive global memory matrix multiplication kernel
__global__ void matMulNaive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float val = 0.0f;
        for (int k = 0; k < N; ++k) {
            val += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = val;
    }
}

// Optimized tiled matrix multiplication kernel using shared memory
__global__ void matMulTiled(const float* A, const float* B, float* C, int N) {
    const int TILE_SIZE = 16;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        if (row < N && m * TILE_SIZE + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + m * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && m * TILE_SIZE + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(m * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            value += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = value;
}

// Utility to initialize matrix with random floats
void initMatrix(std::vector<float>& mat, int N) {
    for (int i = 0; i < N * N; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Function to run a kernel and measure GFLOPS
float runKernelAndMeasure(const char* kernelName,
                          void (*kernel)(const float*, const float*, float*, int),
                          const float* d_A, const float* d_B, float* d_C, int N) {
    int blockSize = 16;
    dim3 threads(blockSize, blockSize);
    dim3 blocks((N + blockSize - 1) / blockSize,
                (N + blockSize - 1) / blockSize);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // Launch kernel
    if (kernelName == std::string("Naive")) {
        matMulNaive<<<blocks, threads>>>(d_A, d_B, d_C, N);
    } else {
        matMulTiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float seconds = milliseconds / 1000.0f;

    // GFLOPS calculation: 2 * N^3 operations
    float gflops = (2.0f * N * N * N) / (seconds * 1e9f);

    // Cleanup events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return gflops;
}

int main() {
    std::vector<int> sizes = {256, 512, 1024, 2048};
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Matrix Size | Naive GFLOPS | Tiled GFLOPS\n";
    std::cout << "-------------------------------------------\n";

    for (int N : sizes) {
        size_t bytes = N * N * sizeof(float);

        // Allocate host memory
        std::vector<float> h_A(N * N);
        std::vector<float> h_B(N * N);
        std::vector<float> h_C(N * N);

        initMatrix(h_A, N);
        initMatrix(h_B, N);

        // Allocate device memory
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

        // Run naive kernel
        float gflopsNaive = runKernelAndMeasure("Naive", matMulNaive, d_A, d_B, d_C, N);

        // Run tiled kernel
        float gflopsTiled = runKernelAndMeasure("Tiled", matMulTiled, d_A, d_B, d_C, N);

        // Copy result back (optional)
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

        // Print results
        std::cout << std::setw(11) << N << " | "
                  << std::setw(12) << gflopsNaive << " | "
                  << std::setw(12) << gflopsTiled << "\n";

        // Free device memory
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    return 0;
}
```