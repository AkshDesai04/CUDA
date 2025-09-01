```cpp
/*
Aim of the program:
Implement `C[i] = pow(A[i], 2.5f)`.

Thinking process:
- The goal is to compute each element of array C as A[i] raised to the power of 2.5.
- In CUDA we can parallelize this by launching one thread per array element.
- We'll use the single-precision floating point function powf for performance.
- The host code will:
  1. Allocate and initialize array A on the host.
  2. Allocate device memory for A and C.
  3. Copy A to the device.
  4. Launch the kernel with enough blocks/threads to cover the array.
  5. Copy the result C back to the host.
  6. Optionally print a few results for verification.
- CUDA error checking will be added via a helper macro to simplify debugging.
- We'll set a constant exponent of 2.5f as requested.
- The code will be self-contained in a single .cu file, ready to compile with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// CUDA kernel to compute C[i] = pow(A[i], 2.5f)
__global__ void powKernel(const float* __restrict__ A, float* __restrict__ C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = powf(A[idx], 2.5f);
    }
}

int main(void) {
    const int N = 1 << 20;           // Number of elements (e.g., 1M)
    const size_t size = N * sizeof(float);

    // Host allocations
    float *h_A = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    if (!h_A || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array A with some values
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i / N;   // values between 0 and 1
    }

    // Device allocations
    float *d_A = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    powKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Simple verification: print first 10 results
    printf("First 10 results of C[i] = A[i]^2.5:\n");
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %.6f\n", i, h_C[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_C);

    printf("Computation completed successfully.\n");
    return EXIT_SUCCESS;
}
```