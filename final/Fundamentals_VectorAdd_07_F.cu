/*
Handle large vectors with boundary checks.

This program demonstrates how to perform element-wise addition of two large vectors on the GPU using CUDA. The vectors are of arbitrary size specified by the user (default 1<<20). The kernel includes boundary checks to ensure that each thread operates only on valid indices. The program allocates memory on the host and device, initializes input vectors, copies them to the device, launches the kernel with an appropriate grid configuration, and then copies the result back to the host. Finally, it verifies the correctness of the operation by comparing a few sample elements and reports the time taken. The code uses C and CUDA and includes basic error handling for CUDA API calls.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Kernel: element-wise vector addition with boundary check
__global__ void vectorAdd(const float *A, const float *B, float *C, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char *argv[]) {
    size_t N = 1 << 20; // Default 1,048,576 elements (~4 MB per array)
    if (argc > 1) {
        N = strtoull(argv[1], NULL, 10);
        if (N == 0) {
            fprintf(stderr, "Invalid vector size provided.\n");
            return EXIT_FAILURE;
        }
    }
    printf("Vector size: %zu elements\n", N);

    // Allocate host memory
    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_B = (float *)malloc(N * sizeof(float));
    float *h_C = (float *)malloc(N * sizeof(float));
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_C, N * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Determine block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (int)((N + threadsPerBlock - 1) / threadsPerBlock);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record the start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    // Record the stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Compute elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few elements
    int errors = 0;
    for (int i = 0; i < 5; ++i) {
        float expected = h_A[i] + h_B[i];
        if (h_C[i] != expected) {
            fprintf(stderr, "Error at index %d: expected %f, got %f\n", i, expected, h_C[i]);
            errors++;
        }
    }
    // Check last element
    size_t last = N - 1;
    float expected_last = h_A[last] + h_B[last];
    if (h_C[last] != expected_last) {
        fprintf(stderr, "Error at last index %zu: expected %f, got %f\n", last, expected_last, h_C[last]);
        errors++;
    }
    if (errors == 0) {
        printf("Result verification passed.\n");
    } else {
        printf("Result verification failed with %d errors.\n", errors);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}
