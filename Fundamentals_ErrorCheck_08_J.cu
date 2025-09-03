/*
Aim: Write a conclusion in comments: always use `cuda-memcheck` during development, and have robust error checking in your code.

Thinking:
- Create a small but complete CUDA example that showcases robust error handling.
- Define a macro `CUDA_CHECK` that wraps every CUDA runtime call, prints an informative message on failure, and exits.
- Implement a simple element‑wise addition kernel.
- Allocate host and device memory, copy data, launch the kernel, copy results back, and clean up.
- Add a final comment that reminds developers to run `cuda-memcheck` and to keep error checking throughout the code.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro to check CUDA runtime API errors
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d - %s (%d)\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err), err);      \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

// Simple kernel that adds two vectors
__global__ void vecAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 1 << 20;          // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    int errors = 0;
    for (int i = 0; i < 10; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            ++errors;
            fprintf(stderr, "Mismatch at index %d: got %f, expected %f\n", i, h_C[i], expected);
        }
    }
    if (errors == 0) {
        printf("Vector addition verified successfully.\n");
    } else {
        printf("Vector addition had %d mismatches.\n", errors);
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

// Conclusion: always use `cuda-memcheck` during development, and have robust error checking in your code.