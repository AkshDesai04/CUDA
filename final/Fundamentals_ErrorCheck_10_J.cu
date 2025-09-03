/*
Use the timer in a loop to average the kernel execution time over many runs.

The aim of this program is to measure the execution time of a CUDA kernel
repeatedly, using CUDA events as timers, and then compute the average
execution time over a number of runs.  This is a typical benchmarking
pattern for CUDA kernels.  The program defines a simple vector addition
kernel, allocates device and host memory, and then runs the kernel in a
loop (e.g., 1000 iterations).  For each iteration, we record the start
and stop events, synchronize, and obtain the elapsed time in milliseconds.
The elapsed times are accumulated and divided by the number of runs to
obtain the average.  The result is printed to stdout.

The code includes basic error checking for CUDA API calls, and verifies
the correctness of the kernel by comparing the device output with a
CPU-computed reference.

Key steps:
1. Define kernel (vector addition).
2. Allocate host and device arrays.
3. Initialize host arrays.
4. Copy data to device.
5. Create CUDA events for timing.
6. Run the kernel in a loop, timing each execution.
7. Accumulate elapsed times and compute average.
8. Copy result back to host and verify.
9. Clean up resources.
10. Print average execution time.

This pattern can be adapted to any kernel and any number of iterations.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel: simple vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Helper macro for checking CUDA errors
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                   \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main(void) {
    // Problem size
    const int N = 1 << 20; // 1M elements
    const int bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C || !h_ref) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
        h_ref[i] = h_A[i] + h_B[i]; // Reference result
    }

    // Allocate device memory
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Determine block and grid sizes
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Number of kernel runs for averaging
    const int numRuns = 1000;
    float totalTimeMs = 0.0f;

    // Warm-up run (optional)
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing loop
    for (int run = 0; run < numRuns; ++run) {
        CHECK_CUDA(cudaEventRecord(start, 0));
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsedMs = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
        totalTimeMs += elapsedMs;
    }

    float avgTimeMs = totalTimeMs / numRuns;
    printf("Average kernel execution time over %d runs: %f ms\n", numRuns, avgTimeMs);

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify results
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - h_ref[i]) > 1e-5f) {
            if (errors < 10) {
                printf("Mismatch at index %d: host=%f, device=%f, ref=%f\n",
                       i, h_C[i], h_ref[i], h_ref[i]);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("Result verification: PASSED\n");
    } else {
        printf("Result verification: FAILED (%d mismatches)\n", errors);
    }

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);

    return 0;
}
