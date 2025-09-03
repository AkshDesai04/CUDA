```cu
/*
Aim of the program:
Profile and compare the naive vs. tiled matrix multiplication kernels using CUDA events. Measure the execution time for a large matrix (e.g., 1024x1024) and report the speedup achieved by using shared memory.

Thinking:
- Use a fixed matrix size of 1024x1024 for reproducibility.
- Allocate pinned host memory for faster transfers (optional but helps).
- Create device arrays for A, B, C_naive, C_tiled.
- Initialize A and B with deterministic values (e.g., A[i][j] = i, B[i][j] = j) to avoid randomness.
- Define naive kernel: each thread computes C[i][j] by looping over k. No shared memory.
- Define tiled kernel: use shared memory to load tiles of A and B, each thread computes a partial sum across tiles.
- Use block size of 16x16 (safe for most GPUs).
- Use cudaEvent_t to time kernel launches: record before and after kernel, then cudaEventElapsedTime.
- Compare elapsed times, compute speedup = naive / tiled.
- Copy results back and optionally verify correctness by comparing arrays element-wise.
- Print times and speedup.
- Free device and host memory.

Edge cases:
- Handle N not divisible by blockSize using ceil division.
- Use proper bounds checks inside kernels.
- Check CUDA errors after API calls.

*/

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << #call << " at line "           \
                      << __LINE__ << ": " << cudaGetErrorString(err)        \
                      << std::endl;                                         \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    }

const int BLOCK_SIZE = 16;

// Naive matrix multiplication kernel
__global__ void matMulNaive(const float *A, const float *B, float *C, int N) {
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

// Tiled matrix multiplication kernel using shared memory
__global__ void matMulTiled(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    int tiles = (N + blockDim.x - 1) / blockDim.x;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int t = 0; t < tiles; ++t) {
        int tiledRow = row;
        int tiledCol = t * blockDim.x + threadIdx.x;
        if (tiledRow < N && tiledCol < N)
            As[threadIdx.y][threadIdx.x] = A[tiledRow * N + tiledCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        tiledRow = t * blockDim.y + threadIdx.y;
        tiledCol = col;
        if (tiledRow < N && tiledCol < N)
            Bs[threadIdx.y][threadIdx.x] = B[tiledRow * N + tiledCol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < blockDim.x; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 1024;
    const size_t size = N * N * sizeof(float);

    // Allocate pinned host memory
    float *h_A, *h_B, *h_C_naive, *h_C_tiled;
    CHECK_CUDA(cudaMallocHost((void **)&h_A, size));
    CHECK_CUDA(cudaMallocHost((void **)&h_B, size));
    CHECK_CUDA(cudaMallocHost((void **)&h_C_naive, size));
    CHECK_CUDA(cudaMallocHost((void **)&h_C_tiled, size));

    // Initialize matrices
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A[i * N + j] = static_cast<float>(i);
            h_B[i * N + j] = static_cast<float>(j);
        }
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C_naive, *d_C_tiled;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C_naive, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C_tiled, size));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Define execution configuration
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Measure naive kernel
    CHECK_CUDA(cudaEventRecord(start));
    matMulNaive<<<grid, block>>>(d_A, d_B, d_C_naive, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_naive = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_naive, start, stop));

    // Measure tiled kernel
    CHECK_CUDA(cudaEventRecord(start));
    matMulTiled<<<grid, block>>>(d_A, d_B, d_C_tiled, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_tiled = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_tiled, start, stop));

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_C_naive, d_C_naive, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_tiled, d_C_tiled, size, cudaMemcpyDeviceToHost));

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < N * N; ++i) {
        if (std::fabs(h_C_naive[i] - h_C_tiled[i]) > 1e-4f) {
            correct = false;
            break;
        }
    }

    std::cout << "Correctness check: " << (correct ? "PASS" : "FAIL") << std::endl;
    std::cout << "Naive kernel time: " << ms_naive << " ms" << std::endl;
    std::cout << "Tiled kernel time: " << ms_tiled << " ms" << std::endl;
    if (ms_tiled > 0.0f) {
        std::cout << "Speedup: " << ms_naive / ms_tiled << "x" << std::endl;
    } else {
        std::cout << "Tiled kernel time too small to compute speedup." << std::endl;
    }

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C_naive));
    CHECK_CUDA(cudaFree(d_C_tiled));
    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_B));
    CHECK_CUDA(cudaFreeHost(h_C_naive));
    CHECK_CUDA(cudaFreeHost(h_C_tiled));

    return 0;
}
```