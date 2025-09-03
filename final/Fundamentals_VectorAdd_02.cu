```cuda
/*
Aim of the program:
Generalize the vector addition program to handle vectors of a large size, e.g., 1,048,576 elements. This will require launching multiple blocks. Each thread must calculate its unique global index using `blockIdx.x * blockDim.x + threadIdx.x` to work on the correct element.

Thinking:
- We need a CUDA kernel that performs element-wise addition: C[i] = A[i] + B[i].
- To handle a large vector (1,048,576 elements), we will launch enough blocks such that each thread processes one element. 
- The kernel will compute the global index: `int idx = blockIdx.x * blockDim.x + threadIdx.x;`.
- We must guard against threads that compute an index beyond the array size: `if (idx < N)`.
- The host code will:
  1. Allocate host vectors A, B, and C with size N.
  2. Initialize A and B with some values (e.g., A[i] = i, B[i] = 2*i).
  3. Allocate device memory for d_A, d_B, d_C.
  4. Copy A and B to device.
  5. Define block size (e.g., 256) and grid size as `(N + blockSize - 1) / blockSize`.
  6. Launch the kernel.
  7. Copy result back to host and verify correctness.
  8. Free device and host memory.
- We'll include simple error checking via a macro `CUDA_CHECK`.
- For demonstration, we'll print the first 10 elements of the result to verify correctness.
*/

#include <stdio.h>
#include <stdlib.h>

#define N 1048576  // 1,048,576 elements
#define BLOCK_SIZE 256

// Simple CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",              \
                    __FILE__, __LINE__, err, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // unique global index
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    size_t bytes = N * sizeof(float);
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = 2.0f * (float)i;
    }

    // Device pointers
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    // Copy host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Determine grid size
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch kernel
    vectorAdd<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("First 10 results of vector addition (A[i] + B[i]):\n");
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```