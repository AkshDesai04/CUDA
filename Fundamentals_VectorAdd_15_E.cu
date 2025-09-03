```cuda
/*
Aim of the program: Fuse a multiply and an add: C[i] = A[i] * scalar1; D[i] = B[i] + scalar2;

Thinking:
1. The program demonstrates a simple CUDA kernel that performs two element‑wise operations in a single launch:
   - Multiply each element of array A by scalar1 and store in array C.
   - Add scalar2 to each element of array B and store in array D.
2. Memory management steps:
   - Allocate host memory for arrays A, B, C, D.
   - Initialize A and B with some test values (e.g., A[i] = i, B[i] = i*2).
   - Allocate device memory for A_d, B_d, C_d, D_d.
   - Copy host arrays A and B to device.
3. Kernel design:
   - Compute global thread index.
   - Guard against out‑of‑bounds indices.
   - Perform the two operations and write results to C_d and D_d.
4. After kernel execution:
   - Copy C_d and D_d back to host arrays C and D.
   - Verify by printing a few sample values.
   - Clean up device and host memory.
5. Additional niceties:
   - Use a simple error‑checking macro to validate CUDA API calls.
   - Use a reasonably sized array (e.g., 1<<20 elements) for demonstration.
   - Keep the program self‑contained and compilable with `nvcc`.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel that fuses a multiply and an add
__global__ void fuseMulAdd(const float *A, const float *B,
                           float *C, float *D,
                           float scalar1, float scalar2,
                           size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] * scalar1;
        D[idx] = B[idx] + scalar2;
    }
}

int main(void)
{
    const size_t N = 1 << 20;          // 1,048,576 elements
    const size_t bytes = N * sizeof(float);

    // Scalars
    const float scalar1 = 2.5f;
    const float scalar2 = 3.0f;

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    float *h_D = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C || !h_D) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = 2.0f * (float)i;
    }

    // Allocate device memory
    float *d_A = NULL, *d_B = NULL, *d_C = NULL, *d_D = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_D, bytes));

    // Copy input data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (int)((N + threadsPerBlock - 1) / threadsPerBlock);

    fuseMulAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_D,
                                                  scalar1, scalar2, N);
    // Check for launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost));

    // Simple verification: print first 5 elements
    printf("First 5 results:\n");
    for (int i = 0; i < 5; ++i) {
        printf("A[%d] = %f, B[%d] = %f => C[%d] = %f, D[%d] = %f\n",
               i, h_A[i], i, h_B[i], i, h_C[i], i, h_D[i]);
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

    printf("Program completed successfully.\n");
    return EXIT_SUCCESS;
}
```