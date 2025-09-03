/*
Implement for `double` data type.
This program demonstrates a simple CUDA kernel that performs element‑wise addition of two
vectors of type double. The main points covered are:

1. **Kernel definition** – a __global__ function that takes pointers to device memory
   and the vector length. Each thread adds the corresponding elements from the
   input arrays and writes the result to the output array.
2. **Memory allocation** – host vectors are allocated with malloc; device vectors
   are allocated with cudaMalloc. The size is defined by the macro N.
3. **Memory transfer** – cudaMemcpy is used to copy the host vectors to device
   memory before launching the kernel, and to copy the result back to the host
   afterwards.
4. **Launch configuration** – the number of blocks and threads per block is
   chosen so that every element is processed by a thread. For N elements we
   compute the number of blocks as (N + blockSize - 1) / blockSize.
5. **Error checking** – a helper macro CHECK_CUDA is defined to wrap CUDA API
   calls and print errors with the file and line number if any call fails.
6. **Performance measurement** – cudaEvent_t objects are used to time the
   kernel execution only (excluding memory copy overhead).
7. **Result verification** – after the kernel finishes, the host checks that
   each element of the result vector equals the sum of the corresponding
   elements from the input vectors.

The program compiles with `nvcc` and runs on a system with a CUDA‑capable
GPU. No external libraries are required beyond the CUDA runtime. */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1<<20)               // Size of vectors (1 million elements)
#define BLOCK_SIZE 256          // Threads per block

/* Macro for checking CUDA API errors */
#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",      \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

/* Kernel: vector addition for double */
__global__ void vecAdd(const double *A, const double *B, double *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    /* Host allocations */
    double *h_A = (double*)malloc(N * sizeof(double));
    double *h_B = (double*)malloc(N * sizeof(double));
    double *h_C = (double*)malloc(N * sizeof(double));

    /* Initialize input vectors */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (double)i;
        h_B[i] = (double)(N - i);
    }

    /* Device allocations */
    double *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, N * sizeof(double)));

    /* Copy inputs to device */
    CHECK_CUDA(cudaMemcpy(d_A, h_A, N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, N * sizeof(double), cudaMemcpyHostToDevice));

    /* Launch kernel */
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    vecAdd<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, N);

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    /* Measure elapsed time */
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel execution time: %f ms\n", ms);

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_C, d_C, N * sizeof(double), cudaMemcpyDeviceToHost));

    /* Verify results */
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        double expected = h_A[i] + h_B[i];
        if (h_C[i] != expected) {
            if (errors < 10) {
                printf("Mismatch at index %d: got %f, expected %f\n",
                       i, h_C[i], expected);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("All results are correct.\n");
    } else {
        printf("Total mismatches: %d\n", errors);
    }

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
