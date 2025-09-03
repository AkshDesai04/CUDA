/*
Analyze the instruction mix reported by the profiler. The tiled version should have a higher ratio of arithmetic instructions to memory instructions.

The goal of this program is to provide two implementations of a matrix multiplication (C = A*B): a naive implementation that accesses global memory directly, and a tiled implementation that uses shared memory to reduce global memory traffic. By running both kernels under the CUDA profiler (e.g., nvprof, Nsight Compute, or Nsight Systems), one can observe the instruction mix for each version. The tiled kernel should exhibit a higher proportion of arithmetic instructions relative to memory instructions because it accesses shared memory for repeated use of A and B elements, thereby reducing the number of global memory loads. The program also includes basic timing using cudaEvent to compare runtime performance, though the primary focus is on instruction mix analysis. The code is written in CUDA C and is self-contained; compiling with nvcc and executing will allow you to profile each kernel separately.

The main steps of the implementation:
1. Allocate host matrices A, B, and result matrices C_naive and C_tiled.
2. Initialize A and B with sample values (e.g., sequential or random).
3. Allocate device memory for A, B, and C.
4. Copy input matrices to device.
5. Launch naive kernel and tiled kernel with identical grid and block dimensions.
6. Copy results back to host.
7. Verify correctness by comparing C_naive and C_tiled (optional).
8. Print timing information for each kernel.
9. Clean up device and host memory.

Key points for instruction mix:
- In the naive kernel, each thread reads two values from global memory per element of C (one from A, one from B) for each iteration of the inner loop; these loads dominate the memory instruction count.
- In the tiled kernel, each block loads a tile of A and B into shared memory once; subsequent uses of those elements come from shared memory, which are considered register accesses, thus the memory instruction count is lower and arithmetic instructions (multiplications/additions) remain the same, leading to a higher arithmetic-to-memory ratio.

Compile with: nvcc -o matmul matmul.cu
Run: ./matmul

Profile with:
- nvprof ./matmul
- nvidia-smi for GPU stats
- Nsight Compute or Nsight Systems for detailed instruction mix.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 512            // Matrix dimension (must be a multiple of TILE_SIZE for simplicity)
#define TILE_SIZE 16     // Tile size for shared memory

// Naive matrix multiplication kernel
__global__ void matMulNaive(const float *A, const float *B, float *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
        {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Tiled matrix multiplication kernel using shared memory
__global__ void matMulTiled(const float *A, const float *B, float *C, int n)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t)
    {
        // Load elements into shared memory
        int A_row = row;
        int A_col = t * TILE_SIZE + threadIdx.x;
        if (A_row < n && A_col < n)
            As[threadIdx.y][threadIdx.x] = A[A_row * n + A_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        int B_row = t * TILE_SIZE + threadIdx.y;
        int B_col = col;
        if (B_row < n && B_col < n)
            Bs[threadIdx.y][threadIdx.x] = B[B_row * n + B_col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product for this tile
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n)
    {
        C[row * n + col] = sum;
    }
}

// Utility to check CUDA errors
void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s (%s)\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Host function to verify that two matrices are equal within a tolerance
bool verify(const float *A, const float *B, int n)
{
    const float eps = 1e-3f;
    for (int i = 0; i < n * n; ++i)
    {
        if (fabs(A[i] - B[i]) > eps)
        {
            printf("Mismatch at index %d: %f vs %f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_naive = (float *)malloc(bytes);
    float *h_C_tiled = (float *)malloc(bytes);

    // Initialize matrices A and B with some values
    for (int i = 0; i < N * N; ++i)
    {
        h_A[i] = static_cast<float>(i % 100);
        h_B[i] = static_cast<float>((i * 2) % 100);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C_naive, *d_C_tiled;
    checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");
    checkCuda(cudaMalloc(&d_B, bytes), "cudaMalloc d_B");
    checkCuda(cudaMalloc(&d_C_naive, bytes), "cudaMalloc d_C_naive");
    checkCuda(cudaMalloc(&d_C_tiled, bytes), "cudaMalloc d_C_tiled");

    // Copy data to device
    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "cudaMemcpy A");
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "cudaMemcpy B");

    // Define block and grid dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Events for timing
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    // Launch naive kernel
    checkCuda(cudaEventRecord(start), "cudaEventRecord start naive");
    matMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C_naive, N);
    checkCuda(cudaEventRecord(stop), "cudaEventRecord stop naive");
    checkCuda(cudaGetLastError(), "Kernel launch error naive");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop naive");

    float msNaive = 0.0f;
    checkCuda(cudaEventElapsedTime(&msNaive, start, stop), "cudaEventElapsedTime naive");

    // Launch tiled kernel
    checkCuda(cudaEventRecord(start), "cudaEventRecord start tiled");
    matMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C_tiled, N);
    checkCuda(cudaEventRecord(stop), "cudaEventRecord stop tiled");
    checkCuda(cudaGetLastError(), "Kernel launch error tiled");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop tiled");

    float msTiled = 0.0f;
    checkCuda(cudaEventElapsedTime(&msTiled, start, stop), "cudaEventElapsedTime tiled");

    // Copy results back to host
    checkCuda(cudaMemcpy(h_C_naive, d_C_naive, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy d_C_naive to h");
    checkCuda(cudaMemcpy(h_C_tiled, d_C_tiled, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy d_C_tiled to h");

    // Verify results
    bool ok = verify(h_C_naive, h_C_tiled, N);
    printf("Verification: %s\n", ok ? "PASS" : "FAIL");

    // Print timing
    printf("Naive kernel time: %.3f ms\n", msNaive);
    printf("Tiled kernel time: %.3f ms\n", msTiled);

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_tiled);
    checkCuda(cudaFree(d_A), "cudaFree d_A");
    checkCuda(cudaFree(d_B), "cudaFree d_B");
    checkCuda(cudaFree(d_C_naive), "cudaFree d_C_naive");
    checkCuda(cudaFree(d_C_tiled), "cudaFree d_C_tiled");
    checkCuda(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy stop");

    return 0;
}
