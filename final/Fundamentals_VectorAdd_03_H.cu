```c
/*
Aim of the program:
Write a main function that tests a range of N values, some multiples of the block size and some not, and verifies all of them.

Thinking:
- The goal is to demonstrate a CUDA kernel works correctly for various input sizes.
- We'll choose a block size of 256 threads, a common value for GPUs.
- We'll test a list of N values that includes:
  * N that is a multiple of 256 (e.g., 256, 512, 1024)
  * N that is not a multiple of 256 (e.g., 100, 300, 600, 1234, 5000, 8192)
- The kernel will simply increment each element of an array by 1. This is easy to verify.
- For each N:
  * Allocate a host array and initialize it with known values.
  * Allocate a device array, copy the data from host to device.
  * Determine grid and block dimensions: gridDim.x = (N + BLOCK_SIZE - 1) / BLOCK_SIZE.
  * Launch the kernel.
  * Copy the data back to the host.
  * Verify that each element equals its original value + 1.
  * Report success or failure.
- We'll include a simple error-checking macro to wrap CUDA API calls.
- The program is self‑contained and can be compiled with nvcc:
      nvcc -o test_increment test_increment.cu
- No external dependencies beyond the CUDA runtime.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Macro for checking CUDA API errors
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Simple kernel that increments each element by 1
__global__ void increment(int *d_arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_arr[idx] += 1;
    }
}

int main(void) {
    // Test values: some multiples of BLOCK_SIZE, some not
    int testNs[] = {100, 256, 300, 512, 600, 1024, 1234, 2048, 5000, 8192};
    const int numTests = sizeof(testNs) / sizeof(testNs[0]);

    for (int t = 0; t < numTests; ++t) {
        int N = testNs[t];
        printf("Testing N = %d\n", N);

        // Allocate host memory
        int *h_arr = (int*)malloc(N * sizeof(int));
        if (!h_arr) {
            fprintf(stderr, "Failed to allocate host memory for N=%d\n", N);
            exit(EXIT_FAILURE);
        }

        // Initialize host array
        for (int i = 0; i < N; ++i) {
            h_arr[i] = i;
        }

        // Allocate device memory
        int *d_arr;
        CUDA_CHECK(cudaMalloc((void**)&d_arr, N * sizeof(int)));

        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice));

        // Determine grid dimensions
        int threadsPerBlock = BLOCK_SIZE;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        // Launch kernel
        increment<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);
        CUDA_CHECK(cudaGetLastError()); // Check kernel launch errors

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

        // Verify results
        int errors = 0;
        for (int i = 0; i < N; ++i) {
            if (h_arr[i] != i + 1) {
                if (errors < 10) { // Print first few errors
                    fprintf(stderr, "Error at index %d: expected %d, got %d\n",
                            i, i + 1, h_arr[i]);
                }
                errors++;
            }
        }

        if (errors == 0) {
            printf("  [PASS] All %d elements verified.\n", N);
        } else {
            printf("  [FAIL] %d mismatches found for N=%d.\n", errors, N);
        }

        // Clean up
        CUDA_CHECK(cudaFree(d_arr));
        free(h_arr);
    }

    printf("All tests completed.\n");
    return 0;
}
```