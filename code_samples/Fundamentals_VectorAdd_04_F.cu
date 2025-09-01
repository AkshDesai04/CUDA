/*
Aim of the program: Handle large vectors with boundary checks.

Thinking:
The goal is to demonstrate how to safely perform element‑wise operations on very large vectors
in CUDA. The key points are:
- Memory allocation on host and device for large arrays.
- Proper kernel launch configuration (grid/block dimensions) that can handle arbitrary sizes.
- Boundary checks inside the kernel so that each thread only writes to valid memory
  even when the grid size is rounded up to the nearest multiple of the block size.
- Simple error‑checking utilities to capture CUDA runtime errors.
- Verification of a few results on the host to confirm correctness.
The program will perform a vector addition: C = A + B. It will be written in plain C
with CUDA extensions, compiled with `nvcc`. No external libraries are used, making
the code portable and straightforward to test with large data sizes.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* Kernel: element‑wise vector addition with boundary check */
__global__ void vecAddKernel(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char *argv[])
{
    /* Default vector size: 16 million elements (~64 MB for each array) */
    const int defaultSize = 16 * 1024 * 1024;
    int N = defaultSize;
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "Invalid vector size.\n");
            return EXIT_FAILURE;
        }
    }

    size_t bytes = N * sizeof(float);

    /* Host allocations */
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host vectors */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i * 0.5f;
        h_B[i] = (float)(N - i) * 1.0f;
    }

    /* Device allocations */
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Kernel launch configuration */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Wait for GPU to finish */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Verify a few results */
    int errors = 0;
    for (int i = 0; i < 10; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            fprintf(stderr, "Mismatch at index %d: got %f, expected %f\n",
                    i, h_C[i], expected);
            errors++;
        }
    }
    if (errors == 0) {
        printf("Vector addition succeeded for first 10 elements.\n");
    } else {
        printf("Vector addition failed with %d mismatches.\n", errors);
    }

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    /* Reset device (optional) */
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
