/*
Use the timer in a loop to average the kernel execution time over many runs.

The goal of this CUDA program is to demonstrate how to measure and average the execution time of a kernel across many iterations. We will use CUDA events (cudaEvent_t) to record the start and end times of each kernel launch. By running the kernel multiple times (controlled by the user-specified number of iterations), we accumulate the total elapsed time and then compute the average. This approach is common in performance benchmarking where the variability of a single run is high, so averaging gives a more stable estimate.

Key points considered:
- Choosing a simple kernel (vector addition) for clarity.
- Allocating device memory for input and output arrays.
- Initializing host data and copying it to the device.
- Creating CUDA events for timing.
- Looping over the specified number of runs, launching the kernel each time, and synchronizing on the end event.
- Accumulating the elapsed time in milliseconds.
- Calculating and printing the average runtime.
- Cleaning up resources (freeing memory, destroying events).
- Using basic error checking after CUDA API calls for robustness.
- The code is self-contained, compiles with nvcc, and requires no external headers beyond the CUDA runtime.

The user only wants the .cu file content, with the aim and thoughts in a comment block before the actual code. All other supporting information is omitted as per instructions.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* CUDA error checking macro */
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* Simple vector addition kernel */
__global__ void vecAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main(int argc, char *argv[]) {
    /* Default parameters */
    int N = 1 << 20;          /* 1M elements */
    int iterations = 100;     /* Number of kernel launches to average over */

    /* Parse command line arguments if provided */
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) iterations = atoi(argv[2]);

    size_t size = N * sizeof(float);

    /* Host allocations */
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    /* Device allocations */
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size));

    /* Copy data to device */
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* CUDA events for timing */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    /* Kernel launch configuration */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Accumulate elapsed time over iterations */
    float totalTimeMs = 0.0f;

    for (int iter = 0; iter < iterations; ++iter) {
        /* Record start event */
        CHECK_CUDA(cudaEventRecord(start, 0));

        /* Launch kernel */
        vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

        /* Record stop event */
        CHECK_CUDA(cudaEventRecord(stop, 0));

        /* Wait for the kernel to finish */
        CHECK_CUDA(cudaEventSynchronize(stop));

        /* Calculate elapsed time for this iteration */
        float elapsedMs = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
        totalTimeMs += elapsedMs;
    }

    /* Compute average */
    float avgTimeMs = totalTimeMs / static_cast<float>(iterations);

    printf("Kernel executed %d times on %d elements.\n", iterations, N);
    printf("Average execution time: %f ms\n", avgTimeMs);

    /* Optional: copy result back to host to verify correctness */
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    /* Simple verification for first element */
    if (h_C[0] == h_A[0] + h_B[0])
        printf("Result verified for element 0.\n");
    else
        printf("Result mismatch!\n");

    /* Clean up */
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
