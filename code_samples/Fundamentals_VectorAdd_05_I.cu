/*
Time the multiplication kernel.
In order to time the multiplication kernel we need a CUDA program that allocates two matrices on the host, copies them to the device, launches a simple matrix multiplication kernel, and measures the execution time using CUDA events. 
The steps are:
1. Define the matrix size (e.g. 1024x1024) and allocate host memory for A, B, and C.
2. Initialize A and B with some values (e.g. random floats).
3. Allocate device memory for d_A, d_B, d_C.
4. Copy A and B to the device.
5. Create CUDA events for timing (start and stop).
6. Record the start event, launch the kernel, record the stop event.
7. Synchronize and compute elapsed time.
8. Copy the result back to the host (optional, for correctness check).
9. Free all allocated memory and destroy events.
10. Print the elapsed GPU time in milliseconds.

We use a simple naive kernel where each thread computes one element of the output matrix. The kernel uses block size 16x16 threads and grid dimensions computed from the matrix size. Error checking macros are used to simplify CUDA API error handling. The program prints the measured GPU execution time.

No additional output or supporting files are produced; only the .cu source code follows.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024                // Matrix dimension (N x N)
#define BLOCK_SIZE 16         // Block size (BLOCK_SIZE x BLOCK_SIZE)

#define CHECK_CUDA(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

__global__ void matMulKernel(const float *A, const float *B, float *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main(void)
{
    size_t bytes = N * N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host matrices.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host matrices with random values */
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));

    /* Copy host data to device */
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Define execution configuration */
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /* Create CUDA events for timing */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    /* Record start event */
    CHECK_CUDA(cudaEventRecord(start, 0));

    /* Launch kernel */
    matMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Record stop event */
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    /* Compute elapsed time */
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    /* Copy result back to host (optional) */
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Print timing result */
    printf("Matrix multiplication kernel execution time: %.3f ms\n", milliseconds);

    /* Clean up */
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
