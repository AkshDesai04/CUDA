/*
 * Aim: Implement for `double` data type.
 *
 * Thinking:
 *   The goal is to demonstrate a simple CUDA program that operates on double
 *   precision floating point numbers.  A classic example is vector addition,
 *   where two input arrays A and B of type double are added element‑wise
 *   to produce an output array C.  This example highlights the following
 *   key CUDA concepts:
 *
 *   1. Device memory allocation for double arrays using cudaMalloc.
 *   2. Copying data between host (CPU) and device (GPU) with cudaMemcpy.
 *   3. Launching a CUDA kernel that uses the double type.  The kernel
 *      calculates one element per thread and checks bounds.
 *   4. Synchronizing and checking for errors after the kernel launch.
 *   5. Cleaning up device memory with cudaFree.
 *
 *   Double precision support requires a GPU with compute capability
 *   1.3 or higher.  The code does not explicitly enforce a compute
 *   capability, but it will fail at runtime on GPUs that lack double
 *   precision support.  If needed, compile with a flag such as
 *   -arch=sm_20 to target a GPU that supports double precision.
 *
 *   The program also includes basic performance timing using CUDA
 *   events, which provides insight into kernel execution time.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define CHECK_CUDA(call)                                                    \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

__global__ void vectorAdd(const double *A, const double *B, double *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    const int N = 1 << 20;              // 1M elements
    const size_t size = N * sizeof(double);

    // Allocate host memory
    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C = (double *)malloc(size);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0 * i;
        h_B[i] = 2.0 * i;
    }

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify result
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        double expected = h_A[i] + h_B[i];
        if (abs(h_C[i] - expected) > 1e-9) {
            ++errors;
            if (errors <= 10) { // Print first few errors
                printf("Error at index %d: %f + %f = %f, got %f\n",
                       i, h_A[i], h_B[i], expected, h_C[i]);
            }
        }
    }
    if (errors == 0) {
        printf("All results are correct.\n");
    } else {
        printf("Found %d errors.\n", errors);
    }

    // Cleanup
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
