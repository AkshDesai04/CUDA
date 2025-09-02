```cuda
/*
Aim of the program:
Implement a parallel histogram calculation for an array of integers. 
Multiple threads will process the input array and may try to increment the same histogram bin simultaneously. 
Use `atomicAdd` on the bin counters to prevent data races and ensure a correct histogram.

Thinking:
To compute a histogram in parallel we distribute the input elements across CUDA threads. 
Each thread reads an element, determines the corresponding bin index, and increments the bin counter. 
Because multiple threads may target the same bin, we must protect the update with an atomic operation. 
CUDA provides `atomicAdd` for integers which guarantees that the increment is performed atomically, 
eliminating data races. 

The kernel uses a grid‑stride loop so that any number of elements can be processed regardless of the total number of threads. 
The histogram array is stored in global memory and is zeroed before the launch. 
After the kernel completes, we copy the histogram back to the host and print it. 
This code demonstrates correct use of `atomicAdd` and handles allocation, initialization, and cleanup. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define HISTOGRAM_BINS 256          // Number of bins in the histogram
#define THREADS_PER_BLOCK 256       // Number of threads per block

/* Utility macro for checking CUDA errors */
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", \
                    __func__, __FILE__, __LINE__,            \
                    cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

/* Kernel to compute histogram using atomicAdd */
__global__ void histogramKernel(const int *data, int num_elements, int *histogram)
{
    // Compute global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop to cover all elements
    for (int i = tid; i < num_elements; i += stride)
    {
        int val = data[i];
        // Clamp value to histogram range
        if (val < 0) val = 0;
        if (val >= HISTOGRAM_BINS) val = HISTOGRAM_BINS - 1;

        // Atomically increment the appropriate bin
        atomicAdd(&histogram[val], 1);
    }
}

int main(void)
{
    const int NUM_ELEMENTS = 1 << 20;   // 1M elements
    int *h_input = NULL;
    int *d_input = NULL;
    int *d_histogram = NULL;
    int *h_histogram = NULL;

    /* Allocate host memory */
    h_input = (int*)malloc(NUM_ELEMENTS * sizeof(int));
    if (!h_input) {
        fprintf(stderr, "Failed to allocate host input array.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host input with random integers in [0, HISTOGRAM_BINS-1] */
    srand((unsigned)time(NULL));
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        h_input[i] = rand() % HISTOGRAM_BINS;
    }

    /* Allocate device memory */
    CUDA_CHECK(cudaMalloc((void**)&d_input, NUM_ELEMENTS * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_histogram, HISTOGRAM_BINS * sizeof(int)));

    /* Copy input to device */
    CUDA_CHECK(cudaMemcpy(d_input, h_input, NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice));

    /* Initialize histogram on device to zeros */
    CUDA_CHECK(cudaMemset(d_histogram, 0, HISTOGRAM_BINS * sizeof(int)));

    /* Determine grid size */
    int num_blocks = (NUM_ELEMENTS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (num_blocks > 65535) num_blocks = 65535; // CUDA limit

    /* Launch kernel */
    histogramKernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, NUM_ELEMENTS, d_histogram);
    CUDA_CHECK(cudaGetLastError());

    /* Allocate host histogram and copy back */
    h_histogram = (int*)malloc(HISTOGRAM_BINS * sizeof(int));
    if (!h_histogram) {
        fprintf(stderr, "Failed to allocate host histogram array.\n");
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaMemcpy(h_histogram, d_histogram, HISTOGRAM_BINS * sizeof(int), cudaMemcpyDeviceToHost));

    /* Print histogram */
    printf("Histogram (bin : count):\n");
    for (int i = 0; i < HISTOGRAM_BINS; ++i) {
        printf("%3d : %d\n", i, h_histogram[i]);
    }

    /* Clean up */
    free(h_input);
    free(h_histogram);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_histogram));

    return EXIT_SUCCESS;
}
```