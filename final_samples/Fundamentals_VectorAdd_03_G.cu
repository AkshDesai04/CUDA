/*
Apply the boundary check to a kernel that takes two input vectors and one output vector.

The aim of this program is to demonstrate how to correctly apply a boundary check in a CUDA kernel
when operating on two input vectors and producing one output vector.  The boundary check ensures
that each thread only processes a valid index within the bounds of the arrays, preventing out‑of‑bounds
memory accesses which could lead to crashes or incorrect results.  In this example the kernel performs
element‑wise addition of the two input vectors.  The program allocates host memory, initializes
the vectors, copies them to the device, launches the kernel with appropriate grid/block
dimensions, copies the result back to the host, and finally frees all allocated resources.
This minimal example illustrates the core steps involved in writing a CUDA application
with proper boundary checking for vector operations.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N (1 << 20)  // Number of elements (1,048,576)

// CUDA error checking macro
#define CUDA_CHECK(call)                                           \
    {                                                              \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    }

// Kernel that adds two vectors with boundary check
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    float *h_A = NULL;
    float *h_B = NULL;
    float *h_C = NULL;
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    h_A = (float *)malloc(bytes);
    h_B = (float *)malloc(bytes);
    h_C = (float *)malloc(bytes);
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;  // Example value
        h_B[i] = 2.0f;  // Example value
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Determine grid and block sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("First 10 results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Vector addition completed successfully.\n");
    return EXIT_SUCCESS;
}
