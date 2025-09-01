/*
Use `nsight compute` or `nvprof` to compare the global memory traffic between the two versions. The tiled version should have significantly less traffic.

Thinking:
- We need two CUDA kernels: a naive matrix multiplication and a tiled version using shared memory.
- The naive kernel will load elements of A and B from global memory for every computation, leading to high memory traffic.
- The tiled kernel will load sub-matrices into shared memory once per tile, reducing global memory accesses.
- We'll measure execution times with cuda events and encourage the user to run the binary under Nsight Compute or nvprof to observe memory traffic metrics.
- The program will also verify correctness by comparing the two results.
- For simplicity, we will use single precision floats and a square matrix size divisible by the tile width.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024                 // Matrix dimension (NxN)
#define TILE_WIDTH 16          // Tile width for shared memory kernel

// Naive matrix multiplication kernel
__global__ void matMulNaive(const float *A, const float *B, float *C, int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index of C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index of C

    if (row < width && col < width)
    {
        float sum = 0.0f;
        for (int k = 0; k < width; ++k)
        {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Tiled matrix multiplication kernel using shared memory
__global__ void matMulTiled(const float *A, const float *B, float *C, int width)
{
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int m = 0; m < (width + TILE_WIDTH - 1) / TILE_WIDTH; ++m)
    {
        // Load tile from A
        if (row < width && m * TILE_WIDTH + threadIdx.x < width)
            As[threadIdx.y][threadIdx.x] = A[row * width + m * TILE_WIDTH + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from B
        if (col < width && m * TILE_WIDTH + threadIdx.y < width)
            Bs[threadIdx.y][threadIdx.x] = B[(m * TILE_WIDTH + threadIdx.y) * width + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the two tiles together
        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < width && col < width)
        C[row * width + col] = sum;
}

// Utility function for checking CUDA errors
void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s (%s)\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    const int size = N * N;
    const int bytes = size * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_naive = (float *)malloc(bytes);
    float *h_C_tiled = (float *)malloc(bytes);

    // Initialize input matrices with random values
    for (int i = 0; i < size; ++i)
    {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C_naive, *d_C_tiled;
    checkCudaError(cudaMalloc((void **)&d_A, bytes), "alloc d_A");
    checkCudaError(cudaMalloc((void **)&d_B, bytes), "alloc d_B");
    checkCudaError(cudaMalloc((void **)&d_C_naive, bytes), "alloc d_C_naive");
    checkCudaError(cudaMalloc((void **)&d_C_tiled, bytes), "alloc d_C_tiled");

    // Copy input matrices to device
    checkCudaError(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "memcpy h_A->d_A");
    checkCudaError(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "memcpy h_B->d_B");

    // Define execution configuration
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // Timing events
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");

    // Launch naive kernel
    checkCudaError(cudaEventRecord(start), "record start naive");
    matMulNaive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C_naive, N);
    checkCudaError(cudaEventRecord(stop), "record stop naive");
    checkCudaError(cudaEventSynchronize(stop), "synchronize stop naive");

    float time_naive;
    checkCudaError(cudaEventElapsedTime(&time_naive, start, stop), "elapsed naive");
    printf("Naive kernel time: %f ms\n", time_naive);

    // Launch tiled kernel
    checkCudaError(cudaEventRecord(start), "record start tiled");
    matMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C_tiled, N);
    checkCudaError(cudaEventRecord(stop), "record stop tiled");
    checkCudaError(cudaEventSynchronize(stop), "synchronize stop tiled");

    float time_tiled;
    checkCudaError(cudaEventElapsedTime(&time_tiled, start, stop), "elapsed tiled");
    printf("Tiled kernel time: %f ms\n", time_tiled);

    // Copy results back to host
    checkCudaError(cudaMemcpy(h_C_naive, d_C_naive, bytes, cudaMemcpyDeviceToHost), "memcpy d_C_naive->h_C_naive");
    checkCudaError(cudaMemcpy(h_C_tiled, d_C_tiled, bytes, cudaMemcpyDeviceToHost), "memcpy d_C_tiled->h_C_tiled");

    // Verify correctness
    double maxError = 0.0;
    for (int i = 0; i < size; ++i)
    {
        double diff = fabs(h_C_naive[i] - h_C_tiled[i]);
        if (diff > maxError) maxError = diff;
    }
    printf("Maximum difference between naive and tiled: %e\n", maxError);

    // Clean up
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tiled);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_naive); cudaFree(d_C_tiled);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
