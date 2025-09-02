/*
 * Aim: Handle large vectors with boundary checks.
 *
 * Thinking:
 * 1. We need a simple CUDA program that can process vectors that may be larger than the number of available threads.
 * 2. Use a typical kernel pattern: each thread processes one element, but we must check that the global index is within the vector bounds.
 * 3. We'll allocate large vectors on host, copy them to device, launch kernel with enough blocks to cover all elements, perform elementwise addition, and copy back.
 * 4. Include error checking to catch CUDA API errors.
 * 5. Provide a sample output to confirm correctness.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors following a CUDA API call or kernel launch */
#define CHECK_CUDA(call)                                                        \
    {                                                                           \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",        \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),            \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    }

/* Kernel to perform element-wise addition of two large vectors */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    /* Compute global thread index */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Boundary check: ensure we do not access out-of-bounds elements */
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    /* Size of the vectors (large) */
    const int N = 10'000'000;                // 10 million elements
    const size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host vectors */
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size));

    /* Copy data from host to device */
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Determine execution configuration */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());              // Check for launch errors
    CHECK_CUDA(cudaDeviceSynchronize());         // Wait for kernel to finish

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Print a few sample results */
    printf("Sample results (first 5 elements):\n");
    for (int i = 0; i < 5; ++i)
    {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
