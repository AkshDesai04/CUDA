/*
Handle large vectors with boundary checks.

Thinking:
- The goal is to demonstrate processing of large vectors on the GPU while ensuring that each thread performs a boundary check before accessing global memory, thereby preventing out-of-bounds reads/writes.
- We'll implement a simple element-wise addition kernel that adds two input vectors and stores the result in an output vector.
- To handle vectors that may be larger than the total number of available threads, we compute the global thread index and only process elements where the index is less than the vector size (N).
- The kernel will use a standard boundary check: `if (idx < N) { ... }`.
- In the host code, we allocate large vectors using `malloc`, initialize them, copy them to device memory, launch the kernel with a reasonable block and grid size, copy the results back, and verify correctness.
- Error checking macros will be used to capture CUDA API errors.
- The program is self-contained and can be compiled with `nvcc`.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",    \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),         \
                    cudaGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* Kernel: element-wise addition of two vectors with boundary check */
__global__ void vectorAdd(const float *A, const float *B, float *C, size_t N)
{
    /* Compute global index */
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Boundary check to avoid out-of-bounds memory access */
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    /* Size of vectors (e.g., 10 million elements) */
    const size_t N = 10000000;
    const size_t bytes = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    /* Initialize input vectors */
    for (size_t i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));

    /* Copy data from host to device */
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Launch kernel: choose block size and grid size */
    const int BLOCK_SIZE = 256;
    int gridSize = (int)((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    vectorAdd<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);

    /* Check for kernel launch errors */
    CHECK_CUDA(cudaGetLastError());

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Verify results */
    int errors = 0;
    for (size_t i = 0; i < N; ++i)
    {
        float expected = h_A[i] + h_B[i];
        if (fabsf(h_C[i] - expected) > 1e-5)
        {
            if (errors < 10) /* print first few errors */
            {
                fprintf(stderr,
                        "Mismatch at index %zu: GPU=%f, CPU=%f\n",
                        i, h_C[i], expected);
            }
            ++errors;
        }
    }
    if (errors == 0)
        printf("All results are correct.\n");
    else
        printf("Number of mismatches: %d\n", errors);

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
