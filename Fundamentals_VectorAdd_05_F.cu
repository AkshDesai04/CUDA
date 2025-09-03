```cuda
/* 
Aim of the program: Handle large vectors with boundary checks.

Thinking:
The task is to write a CUDA C program that demonstrates how to safely operate on large vectors,
ensuring that every thread checks that its global index is within the bounds of the vector.
This is critical when working with very large data sets that may not align perfectly with
the total number of threads launched. 

Key design decisions:
1. Use a 1D grid of 1D blocks, with block size chosen as a power of two (e.g., 256).
2. Implement a kernel that performs a simple operation (vector addition with a scalar) using a
   grid‑stride loop. This pattern allows each thread to process multiple elements if the
   vector length exceeds the total number of threads.
3. Inside the kernel, before accessing array elements, check that the computed global
   index is less than the vector length. This boundary check prevents out‑of‑bounds memory
   accesses which could lead to undefined behavior or crashes.
4. On the host, allocate large vectors (e.g., 10 million doubles) and copy them to the GPU.
5. After kernel execution, copy the result back and optionally verify a few elements.
6. Include simple error‑checking macros for CUDA API calls.
7. Keep the code self‑contained and compilable with `nvcc`.

The program will:
- Allocate host vectors A and B of size N.
- Initialize them with some values.
- Copy them to device memory.
- Launch `vector_add_scalar` kernel to compute `C[i] = A[i] + B[i] * scalar`.
- Copy result back and print a few samples.
- Clean up memory.

This demonstrates handling of large vectors and the importance of boundary checks in CUDA kernels. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors following a CUDA API call */
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",           \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

/* Kernel: adds two vectors with a scalar multiplier for the second vector
 * C[i] = A[i] + scalar * B[i]
 * Uses grid-stride loop and boundary check for each global index. */
__global__ void vector_add_scalar(const double *A, const double *B, double *C, double scalar, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < N; i += stride) {
        C[i] = A[i] + scalar * B[i];
    }
}

int main(void)
{
    /* Size of the vectors (large) */
    const size_t N = 10'000'000;           /* 10 million elements */
    const double scalar = 2.5;

    /* Allocate host memory */
    double *h_A = (double *)malloc(N * sizeof(double));
    double *h_B = (double *)malloc(N * sizeof(double));
    double *h_C = (double *)malloc(N * sizeof(double));
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host vectors */
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = (double)i;
        h_B[i] = 1.0 / ((double)i + 1.0);
    }

    /* Allocate device memory */
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_B, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_C, N * sizeof(double)));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(double), cudaMemcpyHostToDevice));

    /* Kernel launch parameters */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    vector_add_scalar<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, scalar, N);
    CUDA_CHECK(cudaGetLastError()); /* Check for launch errors */

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(double), cudaMemcpyDeviceToHost));

    /* Verify a few elements */
    printf("Sample results:\n");
    for (size_t i = 0; i < 5; ++i) {
        printf("C[%zu] = %f\n", i, h_C[i]);
    }
    printf("C[%zu] = %f\n", N - 1, h_C[N - 1]);

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return EXIT_SUCCESS;
}
```