```cuda
/*
Aim of the program:
A performance optimization: have each thread compute a private histogram in registers or shared memory first. Then, at the end of the thread, atomically add the private histogram's counts to the global bins. This reduces atomic contention.

Thinking:
- Input array of unsigned char values (0-255). We want to compute a 256-bin histogram.
- Each thread processes a subset of the array using a loop with stride. It maintains a private histogram in local memory (registers). This reduces contention because the thread only writes to its own private array.
- After processing, each thread atomically adds its local histogram counts to the global histogram array on the device. This reduces atomic operations per input element (one per bin per thread, not per element).
- We implement the kernel accordingly. We must be careful with register usage: 256 ints may be large. In practice, this might spill to local memory, but the concept demonstrates the optimization.
- The host code allocates and initializes input, sets up device memory, zeroes the global histogram, launches kernel, copies back and prints results. We include basic error checking.
- We use `atomicAdd` on `unsigned int` which is supported on compute capability >= 2.0.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_BINS 256
#define BLOCK_SIZE 256

// Macro to check CUDA errors
#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", \
                    #call, __FILE__, __LINE__,                \
                    cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

// Kernel: each thread builds a private histogram and then atomically adds to global histogram
__global__ void histogramKernel(const unsigned char *d_input, int N, unsigned int *d_histogram) {
    // Private histogram in local memory (registers)
    unsigned int hist[NUM_BINS] = {0};

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Each thread processes multiple elements with a stride
    for (int i = tid; i < N; i += stride) {
        unsigned char val = d_input[i];
        hist[val]++;  // Increment local bin count
    }

    // After local counting, atomically add to global histogram
    for (int bin = 0; bin < NUM_BINS; ++bin) {
        atomicAdd(&d_histogram[bin], hist[bin]);
    }
}

int main(void) {
    const int N = 1 << 24; // ~16 million elements
    size_t inputSize = N * sizeof(unsigned char);
    size_t histSize = NUM_BINS * sizeof(unsigned int);

    // Allocate host memory
    unsigned char *h_input = (unsigned char *)malloc(inputSize);
    if (!h_input) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host input with random values [0, 255]
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i) {
        h_input[i] = (unsigned char)(rand() % NUM_BINS);
    }

    // Allocate device memory
    unsigned char *d_input = NULL;
    unsigned int *d_histogram = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_input, inputSize));
    CHECK_CUDA(cudaMalloc((void **)&d_histogram, histSize));
    CHECK_CUDA(cudaMemset(d_histogram, 0, histSize));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice));

    // Launch kernel
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    histogramKernel<<<gridSize, BLOCK_SIZE>>>(d_input, N, d_histogram);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy histogram back to host
    unsigned int *h_histogram = (unsigned int *)malloc(histSize);
    if (!h_histogram) {
        fprintf(stderr, "Failed to allocate host histogram.\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaMemcpy(h_histogram, d_histogram, histSize, cudaMemcpyDeviceToHost));

    // Print first 10 bins as a sanity check
    printf("First 10 histogram bins:\n");
    for (int i = 0; i < 10; ++i) {
        printf("Bin %3d: %u\n", i, h_histogram[i]);
    }

    // Clean up
    free(h_input);
    free(h_histogram);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_histogram));

    return 0;
}
```