```c
/*
Aim of the program: Verify the results on the CPU.

Thought process:
1. We need a CUDA program that performs a computation on the GPU and then verifies the results on the CPU. 
2. Vector addition is a simple, common example that is easy to understand and verify. 
3. Steps:
   - Allocate two input arrays (A, B) and one output array (C) on host.
   - Initialize A and B with some values.
   - Allocate corresponding device arrays (d_A, d_B, d_C).
   - Copy A and B from host to device.
   - Launch a kernel that computes C[i] = A[i] + B[i].
   - Copy result C back to host.
   - On the CPU, compute expected results into another array (C_cpu).
   - Compare d_C (now in host C) to C_cpu element-by-element.
   - Print whether verification succeeded or failed.
4. Add error checking for CUDA API calls and kernel launch. 
5. Use a reasonably sized array (e.g., 1 << 20 elements) to make the GPU work but still quick.
6. Use standard CUDA patterns: grid and block dimensions, error macros.
7. Ensure the code compiles with nvcc and runs on typical GPU hardware.

The program below implements this logic. It includes the kernel, host code, error handling, and verification on the CPU.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N (1 << 20)          // Number of elements
#define THREADS_PER_BLOCK 256

// Error checking macro
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    size_t size = N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_C_cpu = (float*)malloc(size);
    if (!h_A || !h_B || !h_C || !h_C_cpu) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA(cudaDeviceSynchronize()); // Wait for GPU to finish

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Compute expected results on CPU
    for (int i = 0; i < N; ++i) {
        h_C_cpu[i] = h_A[i] + h_B[i];
    }

    // Verify
    int mismatch = 0;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - h_C_cpu[i]) > 1e-5f) {
            mismatch = 1;
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n", i, h_C[i], h_C_cpu[i]);
            break;
        }
    }

    if (!mismatch) {
        printf("Verification PASSED: GPU and CPU results match.\n");
    } else {
        printf("Verification FAILED: GPU and CPU results do not match.\n");
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return EXIT_SUCCESS;
}
```