/*
Aim: Implement vector addition for `long long int`.

Thoughts:
- The program demonstrates how to perform element-wise addition of two large vectors
  where each element is of type `long long int` (64-bit signed integer).
- CUDA uses 64‑bit integer types on the device; `long long int` is supported
  for arithmetic and memory operations.
- The host code will:
  1. Allocate and initialize two input vectors `h_A` and `h_B`.
  2. Allocate corresponding device buffers `d_A`, `d_B`, and `d_C`.
  3. Copy `h_A` and `h_B` to the device.
  4. Launch a kernel where each thread computes one element of `C = A + B`.
  5. Copy the result `d_C` back to host into `h_C`.
  6. Verify a few elements to ensure correctness.
  7. Clean up all allocated memory and CUDA resources.
- We use a simple 1D grid of blocks, with 256 threads per block.
- Error checking is performed via a helper macro `CHECK_CUDA_ERROR`.
- Timing is optional; if desired, one could wrap the kernel launch between
  cudaEventCreate/cudaEventRecord and compute elapsed time, but it's omitted
  here to keep the example focused on vector addition.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro to check CUDA API return status */
#define CHECK_CUDA_ERROR(call)                                    \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    #call, __FILE__, __LINE__,                   \
                    cudaGetErrorString(err));                    \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

/* Kernel for vector addition: C[i] = A[i] + B[i] */
__global__ void vectorAdd(const long long int *A,
                          const long long int *B,
                          long long int *C,
                          int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char *argv[]) {
    /* Vector size: default 1M elements, can be overridden by first command-line argument */
    const int N = (argc > 1) ? atoi(argv[1]) : (1 << 20); // 1,048,576 elements
    size_t bytes = N * sizeof(long long int);

    /* Allocate host memory */
    long long int *h_A = (long long int *)malloc(bytes);
    long long int *h_B = (long long int *)malloc(bytes);
    long long int *h_C = (long long int *)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        exit(EXIT_FAILURE);
    }

    /* Initialize host vectors */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (long long int)i;
        h_B[i] = (long long int)(N - i);
    }

    /* Allocate device memory */
    long long int *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, bytes));

    /* Copy host data to device */
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Check for kernel launch errors */
    CHECK_CUDA_ERROR(cudaGetLastError());

    /* Copy result back to host */
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Verify results for first 10 elements */
    printf("Verification of first 10 elements:\n");
    for (int i = 0; i < 10 && i < N; ++i) {
        long long int expected = h_A[i] + h_B[i];
        if (h_C[i] != expected) {
            fprintf(stderr, "Mismatch at index %d: host %lld, device %lld, expected %lld\n",
                    i, h_A[i], h_B[i], h_C[i], expected);
        } else {
            printf("Index %d: %lld + %lld = %lld (OK)\n", i, h_A[i], h_B[i], h_C[i]);
        }
    }

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    /* Reset device (optional) */
    CHECK_CUDA_ERROR(cudaDeviceReset());

    return 0;
}
