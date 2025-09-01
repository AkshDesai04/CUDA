/*
Implement `D[i] = A[i] * B[i] + C[i]` where C is a third input vector.

Thought Process:
- The goal is to perform an element-wise operation on three input vectors (A, B, C) and store the result in a fourth vector D.
- We'll use CUDA to parallelize the computation across many threads.
- The kernel will compute the result for each index i: D[i] = A[i] * B[i] + C[i].
- Host code will:
  1. Allocate and initialize host arrays for A, B, C.
  2. Allocate device memory and transfer the host data to the device.
  3. Launch the kernel with an appropriate grid and block size.
  4. Copy the result back to the host.
  5. (Optional) Verify a few results and clean up memory.
- We will use a simple size of 1,048,576 elements (1 << 20) for demonstration.
- Error checking will be done with a helper macro to keep the code clean.
- We will use float arrays; the logic can be adapted for other types as needed.
- No external dependencies beyond standard CUDA headers and C libraries.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

__global__ void vectorMulAdd(const float *A, const float *B, const float *C,
                             float *D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        D[idx] = A[idx] * B[idx] + C[idx];
    }
}

int main(void) {
    const int N = 1 << 20; // 1,048,576 elements
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    float *h_D = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C || !h_D) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i * 0.001f;
        h_B[i] = (float)(N - i) * 0.001f;
        h_C[i] = 1.0f; // simple constant
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_D;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_D, bytes));

    // Copy host data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vectorMulAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, d_D, N);
    CHECK_CUDA(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("Sample outputs:\n");
    for (int i = 0; i < 10; ++i) {
        printf("D[%d] = %f\n", i, h_D[i]);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_D));

    printf("Computation completed successfully.\n");
    return EXIT_SUCCESS;
}
