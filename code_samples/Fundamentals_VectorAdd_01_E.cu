/*
Aim: Implement vector subtraction: `C[i] = A[i] - B[i]`.

Thinking:
- We'll create a CUDA kernel that performs element‑wise subtraction on two input vectors.
- The host code will allocate host arrays, initialize them with example data,
  transfer them to device memory, launch the kernel, copy back the result,
  and print a few values to verify correctness.
- We will use a typical block size of 256 threads and compute the grid size
  so that all elements are processed.
- Error checking after CUDA API calls will be performed to catch any
  runtime issues.
- The example uses `float` vectors of size 1024, but the code can handle
  any size provided by the user.
- After execution, the program frees all allocated memory.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", \
                    #call, __FILE__, __LINE__,                \
                    cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

// Kernel: C = A - B
__global__ void vec_sub(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] - B[i];
    }
}

int main(void) {
    const int n = 1024;          // Size of vectors
    size_t bytes = n * sizeof(float);

    // Host allocations
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (int i = 0; i < n; ++i) {
        h_A[i] = (float)i;          // Example: A[i] = i
        h_B[i] = (float)(2 * i);    // Example: B[i] = 2*i
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    vec_sub<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());          // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize());     // Ensure kernel finished

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify and print first 10 results
    printf("First 10 results of C = A - B:\n");
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
