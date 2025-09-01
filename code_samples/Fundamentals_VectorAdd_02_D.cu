/*
Aim: Test your generalized kernel with several large `N` values (e.g., 10000, 500000, 2000000).

Thinking:
The purpose of this CUDA program is to verify that a single, reusable kernel can
handle arrays of varying sizes without any modifications to its code.  We use
a simple element‑wise addition kernel as a stand‑in for a more complex,
generalized kernel that could perform any per‑element operation.  The program
creates three test cases with progressively larger `N` values (10,000; 500,000;
and 2,000,000) to ensure that the kernel scales correctly.

For each `N` the program:
1. Allocates host arrays `h_A`, `h_B`, `h_C` and initializes them with
   deterministic values so that the result can be verified.
2. Allocates corresponding device arrays and copies the input data from host
   to device.
3. Launches the kernel with a configurable block size (256 threads per block)
   and a grid size that covers all elements.
4. Measures the execution time using CUDA events.
5. Copies the result back to host and verifies it by computing a checksum
   (sum of all elements of `C`).  Since `C[i] = A[i] + B[i]` with
   `A[i] = i` and `B[i] = 2*i`, the expected sum is `3 * N * (N-1) / 2`.
6. Frees all allocated memory.

The program prints the size `N`, the kernel launch configuration, the measured
execution time, and the computed checksum along with the expected value.
Error checking is performed after every CUDA call to catch and report
misconfigurations or failures promptly.

This demonstrates that the same kernel works correctly for different data
sizes and provides a template for testing any generalized kernel with large
inputs.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* Generalized kernel: element-wise addition */
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    /* List of test N values */
    int N_values[] = {10000, 500000, 2000000};
    int num_tests = sizeof(N_values) / sizeof(N_values[0]);

    /* Block size for kernel launch */
    const int BLOCK_SIZE = 256;

    for (int test = 0; test < num_tests; ++test) {
        int N = N_values[test];
        size_t bytes = N * sizeof(float);

        printf("\n=== Test %d: N = %d ===\n", test + 1, N);

        /* Allocate host memory */
        float *h_A = (float*)malloc(bytes);
        float *h_B = (float*)malloc(bytes);
        float *h_C = (float*)malloc(bytes);
        if (!h_A || !h_B || !h_C) {
            fprintf(stderr, "Failed to allocate host memory.\n");
            exit(EXIT_FAILURE);
        }

        /* Initialize host arrays */
        for (int i = 0; i < N; ++i) {
            h_A[i] = (float)i;          // 0, 1, 2, ...
            h_B[i] = (float)(2 * i);    // 0, 2, 4, ...
        }

        /* Allocate device memory */
        float *d_A = NULL;
        float *d_B = NULL;
        float *d_C = NULL;
        CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

        /* Copy data from host to device */
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

        /* Launch kernel */
        int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        printf("Launching kernel with %d blocks of %d threads each.\n", gridSize, BLOCK_SIZE);

        /* Timing with CUDA events */
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, 0));

        vectorAdd<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("Kernel execution time: %f ms\n", ms);

        /* Copy result back to host */
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

        /* Verify result by computing checksum */
        double checksum = 0.0;
        for (int i = 0; i < N; ++i) {
            checksum += h_C[i];
        }
        /* Expected sum: sum_{i=0}^{N-1} (i + 2i) = 3 * sum_{i=0}^{N-1} i = 3 * N*(N-1)/2 */
        double expected = 3.0 * N * (N - 1) / 2.0;
        printf("Checksum: %f\n", checksum);
        printf("Expected: %f\n", expected);
        printf("Difference: %e\n", fabs(checksum - expected));

        /* Clean up */
        free(h_A);
        free(h_B);
        free(h_C);
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    printf("\nAll tests completed successfully.\n");
    return 0;
}
