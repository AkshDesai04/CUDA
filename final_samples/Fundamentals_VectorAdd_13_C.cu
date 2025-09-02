/*
Aim of the program: Implement `C[i] = (A[i] > 0) ? A[i] : 0` (ReLU activation function).

Thought process:
1. The goal is to demonstrate a simple CUDA kernel that applies the ReLU operation element‑wise on an input array.
2. I will allocate a host array `A` of float values, populate it with some sample data (including negative and positive numbers), then copy it to device memory.
3. The kernel `relu_kernel` will compute `C[i]` as the ReLU of `A[i]`. Each thread will handle one element; the total number of threads will be the size of the array.
4. After kernel execution, the resulting array `C` will be copied back to the host, and I will print a few values to verify correctness.
5. Basic error checking wrappers (`CUDA_CHECK`) will be used to catch any CUDA runtime errors.
6. The code will be self‑contained, compile with `nvcc` to produce an executable, and will only contain a single `.cu` file as requested.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* CUDA error checking macro */
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

/* Kernel implementing the ReLU activation function */
__global__ void relu_kernel(const float *A, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float val = A[idx];
        C[idx] = (val > 0.0f) ? val : 0.0f;
    }
}

int main(void)
{
    /* Size of the array */
    const int N = 1024;
    const size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    if (h_A == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host input array with some values */
    for (int i = 0; i < N; ++i)
    {
        /* Mix positive and negative numbers */
        h_A[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f; /* Range [-10,10] */
    }

    /* Allocate device memory */
    float *d_A = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    /* Copy input data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    /* Launch kernel: one thread per element */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N);

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Verify a few results */
    printf("Index\tInput\tReLU Output\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("%5d\t%6.3f\t%6.3f\n", i, h_A[i], h_C[i]);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_C);

    return EXIT_SUCCESS;
}
