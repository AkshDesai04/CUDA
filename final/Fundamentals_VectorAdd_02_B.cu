```cuda
/*
Aim: Make the vector size N and the block size BLOCK_SIZE configurable (e.g., using const int or #define). Calculate the grid size dynamically.

Thinking:
1. Use #define or const int to allow easy change of the vector length and block size.
2. Calculate the number of blocks as (N + BLOCK_SIZE - 1) / BLOCK_SIZE to ensure all elements are processed.
3. Provide a simple vector addition kernel: C = A + B.
4. Allocate host memory, initialize A and B, copy to device, run kernel, copy result back, verify correctness, and clean up.
5. Include a simple CUDA error-checking macro for robustness.
6. The code should compile as a standalone .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Configurable parameters
const int N = 1 << 20;          // Vector size (e.g., 1 million elements)
const int BLOCK_SIZE = 256;     // Block size (threads per block)

// CUDA error checking macro
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

// Kernel to add two vectors
__global__ void vecAdd(const float* A, const float* B, float* C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

int main(void)
{
    // Host allocations
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Calculate grid size dynamically
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch kernel
    vecAdd<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few elements
    int errors = 0;
    for (int i = 0; i < 10; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            printf("Mismatch at index %d: host=%f, device=%f, expected=%f\n",
                   i, h_C[i], expected, expected);
            errors++;
        }
    }
    if (errors == 0)
        printf("Vector addition successful! Verified first 10 elements.\n");
    else
        printf("Vector addition had %d errors.\n", errors);

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```