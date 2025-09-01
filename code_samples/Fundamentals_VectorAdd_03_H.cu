```cuda
/*
Write a `main` function that tests a range of `N` values, some multiples of the block size and some not, and verifies all of them.

Thinking:
- The goal is to test a kernel over a variety of problem sizes, ensuring that it works correctly regardless of whether the array length is a multiple of the chosen block size.
- We choose a simple kernel that sums the elements of an integer array using atomicAdd. This is straightforward to verify and highlights handling of partial blocks.
- The kernel will be launched with a grid that covers all elements: `gridDim.x = (N + blockSize - 1) / blockSize`. This automatically handles cases where `N` is not a multiple of `blockSize`.
- For each test case, we will:
  1. Allocate and initialize a host array with known values (1..N).
  2. Allocate device memory for the array and copy the data over.
  3. Allocate device memory for the result (unsigned long long) and initialize it to 0.
  4. Launch the kernel.
  5. Copy back the result and compare it to the expected sum `N*(N+1)/2`.
  6. Print whether the test passed.
- We will test several `N` values: small (< block size), exactly a multiple of block size, slightly above a multiple, and large values including one that is not a multiple of block size.
- The block size is set to 256 threads per block, a common choice for many GPUs.
- Error checking for CUDA API calls is included via a helper macro `CUDA_CHECK`.

This program demonstrates a robust testing strategy for kernel launch parameters and validates correct operation across a range of problem sizes.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel: Sum array elements into a single result using atomicAdd
__global__ void sumKernel(const int *data, int N, unsigned long long *result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(result, static_cast<unsigned long long>(data[idx]));
    }
}

int main()
{
    const int blockSize = 256; // Threads per block
    // Test N values: small, exact multiple, just above multiple, large, and non-multiple
    const int testN[] = {1, 100, 256, 257, 512, 1000, 1024, 1000003, 1000000};
    const int numTests = sizeof(testN) / sizeof(testN[0]);

    for (int t = 0; t < numTests; ++t) {
        int N = testN[t];
        printf("Testing N = %d\n", N);

        // Allocate and initialize host array
        int *h_data = (int*)malloc(N * sizeof(int));
        for (int i = 0; i < N; ++i) {
            h_data[i] = i + 1; // Values 1..N
        }

        // Expected sum
        unsigned long long expected = static_cast<unsigned long long>(N) * (N + 1) / 2;

        // Allocate device memory
        int *d_data;
        unsigned long long *d_result;
        CUDA_CHECK(cudaMalloc((void**)&d_data, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&d_result, sizeof(unsigned long long)));

        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(unsigned long long)));

        // Determine grid dimensions
        int gridSize = (N + blockSize - 1) / blockSize;

        // Launch kernel
        sumKernel<<<gridSize, blockSize>>>(d_data, N, d_result);
        CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

        // Copy result back to host
        unsigned long long h_result;
        CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        // Verify result
        if (h_result == expected) {
            printf("PASS: Result = %llu, Expected = %llu\n", h_result, expected);
        } else {
            printf("FAIL: Result = %llu, Expected = %llu\n", h_result, expected);
        }

        // Clean up
        CUDA_CHECK(cudaFree(d_data));
        CUDA_CHECK(cudaFree(d_result));
        free(h_data);

        printf("\n");
    }

    // Reset device before exiting
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
```