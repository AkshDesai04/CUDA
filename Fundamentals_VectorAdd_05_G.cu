/*
Aim of the program: Implement for `float` data.

In order to satisfy the requirement of implementing for float data in a CUDA program, I chose to demonstrate a simple yet fundamental operation: vector addition. This operation is straightforward to implement, showcases the use of CUDA memory management, kernel launch configuration, and basic error handling, and directly involves float data types as requested.

Thinking process:
1. The goal is to create a self‑contained CUDA example that operates on `float` arrays.
2. I decided to write a kernel `addVectors` that takes two input arrays and writes the element‑wise sum to an output array.
3. The host code will allocate three arrays (`h_A`, `h_B`, `h_C`) on the CPU, initialize `h_A` and `h_B` with sample data, allocate corresponding device memory, copy inputs to the GPU, launch the kernel, copy the result back, and finally print the output to verify correctness.
4. Error checking for CUDA API calls is included to ensure robustness.
5. The code is written in plain C/C++ using the CUDA runtime API and can be compiled with `nvcc`.

This program fulfills the requirement of implementing functionality for `float` data while demonstrating typical CUDA programming patterns.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Kernel to add two float vectors: C = A + B */
__global__ void addVectors(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

/* Utility macro for CUDA error checking */
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",           \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

int main(void)
{
    /* Problem size */
    const int N = 1 << 20;           /* 1M elements */
    const size_t bytes = N * sizeof(float);

    /* Host memory allocation */
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    /* Initialize input vectors */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i * 1.0f;
        h_B[i] = (float)(N - i) * 0.5f;
    }

    /* Device memory allocation */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    /* Copy inputs to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Launch configuration */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Verify a few results */
    int errors = 0;
    for (int i = 0; i < 10; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: GPU %f, CPU %f\n",
                    i, h_C[i], expected);
            ++errors;
        } else {
            printf("Index %d: %f + %f = %f\n", i, h_A[i], h_B[i], h_C[i]);
        }
    }
    if (errors == 0)
        printf("Vector addition verified for first 10 elements.\n");
    else
        printf("Vector addition failed with %d errors.\n", errors);

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return EXIT_SUCCESS;
}
