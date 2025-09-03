/*
Implement `C[i] = (A[i] > 0) ? A[i] : 0` (ReLU activation function).
The goal is to show a minimal CUDA program that applies the ReLU
activation to each element of a float array.  The program will:

1. Define a kernel that performs the conditional assignment per thread.
2. Allocate host memory for input and output arrays.
3. Initialize the input array with some sample values (positive, zero,
   and negative) so that the effect of ReLU is visible.
4. Allocate device memory, copy the input to the device.
5. Launch the kernel with a suitable grid/block configuration.
6. Copy the result back to the host and print both input and output
   arrays for verification.
7. Clean up all allocated memory and check for CUDA errors.

The code includes basic error checking via a macro that prints
CUDA runtime errors.  This example is intentionally simple to
concentrate on the ReLU operation rather than advanced CUDA
optimizations or memory management strategies.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* CUDA kernel that applies ReLU to each element */
__global__ void relu_kernel(const float *A, float *C, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = A[idx];
        C[idx] = (val > 0.0f) ? val : 0.0f;
    }
}

int main(void)
{
    const size_t N = 16; /* number of elements */
    const size_t SIZE = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(SIZE);
    float *h_C = (float *)malloc(SIZE);
    if (!h_A || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize input array with a mix of positive, zero, and negative values */
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = (float)((i % 4) - 2); /* generates -2, -1, 0, 1 repeating */
    }

    /* Allocate device memory */
    float *d_A = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_C, SIZE));

    /* Copy input from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice));

    /* Define block and grid sizes */
    const int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N);
    CUDA_CHECK(cudaGetLastError()); /* Check for launch errors */

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, SIZE, cudaMemcpyDeviceToHost));

    /* Print input and output arrays */
    printf("Index\tA[i]\tC[i] (ReLU)\n");
    for (size_t i = 0; i < N; ++i) {
        printf("%zu\t%.2f\t%.2f\n", i, h_A[i], h_C[i]);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_C);

    return EXIT_SUCCESS;
}
