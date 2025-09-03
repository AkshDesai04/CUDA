/*
Aim: Handle input values that are outside the range of the histogram bins.

Thinking:
To build a histogram on the GPU that gracefully handles values that fall outside the defined bin range, we need to define the binning scheme and decide how to treat out‑of‑range values.  
- Choose a fixed minimum (`minVal`) and maximum (`maxVal`) defining the histogram’s valid data range.  
- Decide how many bins (`numBins`) to use; each bin represents an interval of width `binWidth = (maxVal - minVal) / numBins`.  
- For values < `minVal` we count them as **underflow**; for values > `maxVal` we count them as **overflow**.  
- For values inside the range we compute the bin index:  
  `idx = min( (int)((val - minVal) / binWidth), numBins-1 )`  
  (the min is needed to keep values equal to `maxVal` inside the last bin).  
- Since multiple threads may update the same bin simultaneously, use `atomicAdd` to avoid race conditions.  
- Allocate a bin array on the device with size `numBins + 2` (underflow at index 0, real bins at 1..numBins, overflow at `numBins+1`).  
- Copy input data from host to device, launch kernel, copy histogram back to host, and print results.  

The program demonstrates this approach with a simple example: a randomly generated array of 1000 floating point numbers ranging from -5 to 15, a histogram with bins from 0 to 10, and 5 bins.  The kernel will classify each value and increment the appropriate bin counter atomically.  The host then displays the bin counts, as well as the underflow and overflow counts.  This showcases handling out‑of‑range values cleanly on the GPU.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define CHECK_CUDA(call)                                                          \
    {                                                                             \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                  \
        }                                                                         \
    }

// Kernel to build histogram
__global__ void histogramKernel(const float *data, int n, float minVal, float maxVal,
                                 int numBins, int *histBins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = data[idx];

    if (val < minVal) {
        // Underflow bin
        atomicAdd(&histBins[0], 1);
    } else if (val > maxVal) {
        // Overflow bin
        atomicAdd(&histBins[numBins + 1], 1);
    } else {
        // Compute bin index
        float binWidth = (maxVal - minVal) / numBins;
        int binIdx = (int)((val - minVal) / binWidth) + 1; // +1 offset for underflow bin
        // Handle edge case when val == maxVal
        if (binIdx > numBins) binIdx = numBins;
        atomicAdd(&histBins[binIdx], 1);
    }
}

int main(void) {
    const int N = 1000;                 // Number of data points
    const int numBins = 5;              // Number of histogram bins
    const float minVal = 0.0f;          // Minimum value for histogram range
    const float maxVal = 10.0f;         // Maximum value for histogram range

    // Host allocation
    float *h_data = (float *)malloc(N * sizeof(float));
    int   *h_hist = (int *)malloc((numBins + 2) * sizeof(int)); // +2 for underflow and overflow

    if (!h_data || !h_hist) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize random data between -5 and 15
    srand((unsigned int)time(NULL));
    for (int i = 0; i < N; ++i) {
        h_data[i] = -5.0f + 20.0f * ((float)rand() / RAND_MAX);
    }

    // Initialize histogram bins to zero
    for (int i = 0; i < numBins + 2; ++i) {
        h_hist[i] = 0;
    }

    // Device allocation
    float *d_data = NULL;
    int   *d_hist = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_data, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_hist, (numBins + 2) * sizeof(int)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_hist, h_hist, (numBins + 2) * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    histogramKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N, minVal, maxVal,
                                                        numBins, d_hist);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy histogram back to host
    CHECK_CUDA(cudaMemcpy(h_hist, d_hist, (numBins + 2) * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Histogram results:\n");
    printf("Underflow (< %.2f): %d\n", minVal, h_hist[0]);
    for (int i = 0; i < numBins; ++i) {
        float binLow = minVal + i * ((maxVal - minVal) / numBins);
        float binHigh = binLow + ((maxVal - minVal) / numBins);
        printf("Bin %2d [%.2f, %.2f): %d\n", i, binLow, binHigh, h_hist[i + 1]);
    }
    printf("Overflow (> %.2f): %d\n", maxVal, h_hist[numBins + 1]);

    // Clean up
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_hist));
    free(h_data);
    free(h_hist);

    return EXIT_SUCCESS;
}
