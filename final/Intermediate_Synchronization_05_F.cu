/*
 * Aim of the program: Profile the histogram kernel. The performance will be limited by the contention on the atomic operations.
 *
 * Thinking:
 * The goal is to provide a minimal but complete CUDA program that implements a histogram kernel using atomicAdd to update a
 * per-bin counter. Atomic operations serialize when many threads try to update the same memory location. If the number of bins
 * is small relative to the amount of work, many threads will contend for the same bin, causing a performance bottleneck.
 *
 * In this program we:
 * 1. Allocate a large array of random integers on the host.
 * 2. Copy that array to the device.
 * 3. Allocate a device histogram array (NUM_BINS entries) and initialize it to zero.
 * 4. Launch a kernel where each thread processes a subset of the data, determines its bin index (value % NUM_BINS),
 *    and performs atomicAdd to increment the histogram counter.
 * 5. Use CUDA events to time the kernel execution and report elapsed milliseconds.
 * 6. Copy the resulting histogram back to the host and print a few sample bin counts for verification.
 *
 * The histogram kernel is intentionally simple to highlight the contention on the atomic operations. By using a small
 * number of bins (e.g., 16) we force many threads to update the same bins, which will be reflected in the kernel execution
 * time. Users can experiment by changing NUM_BINS or data size to observe how contention affects performance.
 *
 * The code is written to compile with nvcc and does not rely on any external libraries beyond the standard CUDA runtime API.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_BINS 16          // Small number of bins to increase atomic contention
#define BLOCK_SIZE 256
#define ARRAY_SIZE (1 << 24) // ~16 million elements

// CUDA error checking macro
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Histogram kernel: each thread processes multiple elements
__global__ void histogram_kernel(const int *data, int *hist, size_t N)
{
    // Calculate global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // Process elements in a grid-stride loop
    for (size_t i = idx; i < N; i += stride)
    {
        int val = data[i];
        int bin = val % NUM_BINS; // Simple binning
        atomicAdd(&hist[bin], 1);
    }
}

int main()
{
    // Allocate host data
    int *h_data = (int *)malloc(ARRAY_SIZE * sizeof(int));
    if (!h_data)
    {
        fprintf(stderr, "Failed to allocate host data\n");
        return EXIT_FAILURE;
    }

    // Seed random generator
    srand((unsigned)time(NULL));

    // Fill host data with random integers
    for (size_t i = 0; i < ARRAY_SIZE; ++i)
    {
        h_data[i] = rand() % 256; // Values between 0 and 255
    }

    // Allocate device data
    int *d_data = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_data, ARRAY_SIZE * sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate and initialize device histogram
    int *d_hist = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_hist, NUM_BINS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(int)));

    // Determine grid dimensions
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (ARRAY_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch histogram kernel
    histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_hist, ARRAY_SIZE);

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));

    // Wait for kernel to finish
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Compute elapsed time
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("Histogram kernel execution time: %f ms\n", ms);

    // Copy histogram back to host
    int *h_hist = (int *)malloc(NUM_BINS * sizeof(int));
    if (!h_hist)
    {
        fprintf(stderr, "Failed to allocate host histogram\n");
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost));

    // Print a few histogram bin values for verification
    printf("Histogram bins (bin: count):\n");
    for (int i = 0; i < NUM_BINS; ++i)
    {
        printf("%d: %d\n", i, h_hist[i]);
    }

    // Clean up
    free(h_data);
    free(h_hist);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_hist));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
