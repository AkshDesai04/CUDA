```cuda
/*
 * Aim of the program:
 * Compare the execution time for `N=1M` vs `N=2M`. Is it roughly double?
 *
 * Thinking:
 * To answer this question we need a simple CUDA program that runs a
 * computational kernel on arrays of two different sizes: 1 million
 * elements and 2 million elements.  We'll use a straightforward vector
 * addition kernel because its execution time is largely proportional
 * to the number of elements processed.  By measuring the GPU runtime
 * (using CUDA events) for each size we can compare the two times and
 * compute the ratio.  If the ratio is close to 2, then the execution
 * time is roughly double when the problem size doubles.
 *
 * The program performs the following steps:
 * 1. Define two sizes, N1 = 1,000,000 and N2 = 2,000,000.
 * 2. For each size:
 *    a. Allocate host memory and initialize two input arrays.
 *    b. Allocate device memory and copy inputs to the device.
 *    c. Launch a simple vector addition kernel.
 *    d. Measure the kernel execution time using cudaEvent_t.
 *    e. Copy the result back to the host and optionally validate it.
 *    f. Print the measured time.
 * 3. After both runs, compute the ratio of the two times and print it.
 *
 * The output will show the two timings and the ratio, allowing the
 * user to see whether the execution time scales roughly linearly
 * with the input size.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

#define CHECK_CUDA(call)                                                    \
    {                                                                        \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    }

// Simple vector addition kernel
__global__ void vecAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Helper function to run kernel for a given size and return execution time in ms
float runKernel(int N) {
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f; // arbitrary value
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Determine grid and block sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Record start event
    CHECK_CUDA(cudaEventRecord(start));

    // Launch kernel
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop));

    // Wait for the stop event to complete
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Clean up events
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // Copy result back to host (optional validation)
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Optional: Validate result
    bool valid = true;
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            valid = false;
            break;
        }
    }
    if (!valid) {
        fprintf(stderr, "Result validation failed for N=%d\n", N);
    }

    // Free device memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return milliseconds;
}

int main() {
    const int N1 = 1'000'000;  // 1M elements
    const int N2 = 2'000'000;  // 2M elements

    printf("Running kernel with N = %d\n", N1);
    float time1 = runKernel(N1);
    printf("Time for N=%d: %.3f ms\n", N1, time1);

    printf("\nRunning kernel with N = %d\n", N2);
    float time2 = runKernel(N2);
    printf("Time for N=%d: %.3f ms\n", N2, time2);

    float ratio = time2 / time1;
    printf("\nRatio of times (N2 / N1): %.3f\n", ratio);
    printf("Is the execution time roughly double? %s\n",
           (ratio > 1.8f && ratio < 2.2f) ? "Yes" : "No");

    return 0;
}
```