/*
Deliberately corrupt one byte of the data on the host before verification to ensure your verification code works.

Thinking process:

1. Goal: Demonstrate that a verification routine correctly detects a corruption introduced on the host side.
2. Approach:
   - Generate two input arrays (`h_a`, `h_b`) of integers.
   - Compute a reference result on the CPU (`h_expected`) by simple element-wise addition.
   - Allocate device memory for the inputs and output.
   - Copy inputs to the device and launch a kernel that performs element-wise addition, storing the result in `d_c`.
   - Copy the device output back to the host array `h_gpu`.
   - Intentionally corrupt a single byte in `h_gpu` to simulate data corruption.
   - Run a verification loop that compares `h_gpu` with `h_expected`. Any mismatch should be reported.
3. Implementation details:
   - Use integer arrays for simplicity; corruption of a byte will likely change the value of an element.
   - The corruption is performed by casting the integer array pointer to an `unsigned char*` and modifying the first byte.
   - The verification counts mismatches and prints the number; due to corruption we expect at least one mismatch.
4. CUDA specifics:
   - A simple kernel `vecAdd` that adds corresponding elements of two arrays.
   - Error checking macro `CUDA_CHECK` for CUDA API calls.
   - Standard CUDA memory allocation, copy, and kernel launch patterns.
5. Expected output:
   - Number of mismatches reported. Since we corrupted one byte, we expect at least one mismatch, verifying that the check works.

This code is self-contained and can be compiled with `nvcc` to produce an executable that demonstrates the verification mechanism.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 20)   // 1 million elements
#define BLOCK_SIZE 256

// CUDA error checking macro
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

// Simple element-wise addition kernel
__global__ void vecAdd(const int *a, const int *b, int *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void) {
    int *h_a = NULL;
    int *h_b = NULL;
    int *h_expected = NULL;
    int *h_gpu = NULL;

    int *d_a = NULL;
    int *d_b = NULL;
    int *d_c = NULL;

    size_t bytes = N * sizeof(int);

    // Allocate host memory
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_expected = (int*)malloc(bytes);
    h_gpu = (int*)malloc(bytes);

    if (!h_a || !h_b || !h_expected || !h_gpu) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    // Initialize input arrays with deterministic data
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = N - i;
    }

    // Compute reference result on CPU
    for (int i = 0; i < N; ++i) {
        h_expected[i] = h_a[i] + h_b[i];
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vecAdd<<<grid, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_gpu, d_c, bytes, cudaMemcpyDeviceToHost));

    // Deliberately corrupt one byte of the data on the host
    unsigned char *byte_ptr = (unsigned char*)h_gpu;
    byte_ptr[0] ^= 0xFF;  // Flip all bits of the first byte

    // Verification
    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        if (h_gpu[i] != h_expected[i]) {
            mismatches++;
            // Report first few mismatches for debugging
            if (mismatches <= 10) {
                printf("Mismatch at index %d: host=%d, expected=%d\n",
                       i, h_gpu[i], h_expected[i]);
            }
        }
    }

    printf("Total mismatches detected: %d\n", mismatches);

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_expected);
    free(h_gpu);

    return 0;
}
