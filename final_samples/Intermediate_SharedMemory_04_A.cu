/*
Aim: Calculate and report the performance in GFLOPS (billions of floating-point operations per second). The number of FLOPS for matrix multiplication is `2 * N^3`.

Thinking:
1. The program will perform a single GPU matrix multiplication of two N×N matrices using a simple CUDA kernel.
2. Timing will be measured with CUDA events to capture only the kernel execution time.
3. After the kernel completes, the program will compute the total floating-point operations as `2 * N^3` and divide by the elapsed time (converted to seconds) to obtain GFLOPS.
4. The code will accept an optional command-line argument to specify N; otherwise it defaults to 1024.
5. Memory will be allocated on both host and device, and data will be transferred appropriately.
6. Error handling will be added for CUDA API calls.
7. Finally, the computed GFLOPS value will be printed to standard output.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 16

// CUDA kernel for naive matrix multiplication
__global__ void MatMulKernel(const float *A, const float *B, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index of C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // column index of C

    if (row < N && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
        {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Utility macro for error checking
#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

// Simple RNG to fill matrices
void fillRandom(float *data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = (float)rand() / RAND_MAX;
    }
}

int main(int argc, char *argv[])
{
    int N = 1024; // default size
    if (argc >= 2)
    {
        N = atoi(argv[1]);
        if (N <= 0)
        {
            fprintf(stderr, "Invalid matrix size. Using default N=1024.\n");
            N = 1024;
        }
    }

    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Host memory allocation failed.\n");
        return EXIT_FAILURE;
    }

    // Initialize matrices
    srand(0); // deterministic seed
    fillRandom(h_A, N * N);
    fillRandom(h_B, N * N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Configure execution parameters
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up run (optional)
    MatMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Start timing
    CUDA_CHECK(cudaEventRecord(start));

    // Launch kernel
    MatMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Compute elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float seconds = milliseconds / 1000.0f;

    // Copy result back to host (optional)
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Calculate GFLOPS
    double flops = 2.0 * N * N * N;
    double gflops = flops / seconds / 1e9;

    printf("Matrix size: %d x %d\n", N, N);
    printf("Elapsed time: %f ms\n", milliseconds);
    printf("Performance: %f GFLOPS\n", gflops);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
