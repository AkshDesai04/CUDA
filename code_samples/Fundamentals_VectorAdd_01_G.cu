/*
Implement vector-scalar addition: `B[i] = A[i] + 5;` where 5 is a scalar value passed to the kernel.

Thinking process:
1. Determine data type: The problem statement uses integer literal 5, but vector operations are often performed on floats. I chose float for generality; passing 5.0f ensures correct addition for float arrays.
2. Kernel design: Simple device kernel that adds a scalar to each element. The scalar will be passed as an argument to avoid global memory access for a single value.
3. Thread indexing: Use 1D grid of 1D blocks. Each thread handles one element. Compute global index with blockIdx.x * blockDim.x + threadIdx.x. Guard against out-of-bounds indices.
4. Host side:
   - Allocate host vectors A and B of size N (e.g., 1024).
   - Initialize A with some test values (e.g., i * 1.0f).
   - Allocate device memory for dA and dB.
   - Copy A to dA.
   - Launch kernel with enough blocks: (N + threadsPerBlock - 1) / threadsPerBlock.
   - Copy dB back to host B.
   - Verify results by printing a few elements.
   - Free device and host memory.
5. Error checking: Use a helper macro `CUDA_CHECK` to wrap CUDA API calls for debugging.
6. Make sure the code compiles with `nvcc` and runs correctly.
7. Keep the code self‑contained: includes, main, kernel, and helper functions all in one .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",  \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),      \
                    cudaGetErrorString(err));                           \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// Kernel that adds a scalar to each element of the input array
__global__ void vectorScalarAdd(const float *A, float *B, int N, float scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        B[idx] = A[idx] + scalar;
    }
}

int main(void) {
    const int N = 1024;                     // Number of elements
    const int bytes = N * sizeof(float);    // Size in bytes

    // Host allocations
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    if (h_A == NULL || h_B == NULL) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host input vector
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;  // Example values: 0.0, 1.0, 2.0, ...
    }

    // Device allocations
    float *d_A = NULL;
    float *d_B = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel with scalar value 5.0f
    vectorScalarAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N, 5.0f);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));

    // Simple verification: print first 10 results
    printf("First 10 results of B[i] = A[i] + 5:\n");
    for (int i = 0; i < 10; ++i) {
        printf("B[%d] = %f\n", i, h_B[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    free(h_A);
    free(h_B);

    return EXIT_SUCCESS;
}
