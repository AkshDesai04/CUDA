/*
Aim: Handle input values that are outside the range of the histogram bins.

Thinking:
1. Define a histogram with a fixed number of bins and a value range (min_val to max_val).
2. For each input value, compute the bin index using the formula:
      idx = (int)((value - min_val) / bin_width)
   where bin_width = (max_val - min_val) / NUM_BINS.
3. If idx is within [0, NUM_BINS-1], increment that bin atomically.
4. If idx is outside this range (value < min_val or value >= max_val), increment a separate
   out_of_range counter atomically.
5. Use a CUDA kernel to parallelize the histogram computation.
6. Allocate device memory for the data array, histogram bins, and the out-of-range counter.
7. Generate sample data on the host (or read from file) and copy it to the device.
8. Launch the kernel with an appropriate grid/block configuration.
9. Copy the histogram and out-of-range count back to the host.
10. Print the histogram and the out-of-range count.
11. Include error checking for CUDA API calls.
12. Ensure the program is self‑contained and compiles with nvcc to produce a .cu file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define NUM_BINS 10

__global__ void histogram_kernel(const float *data, int *bins, int *out_of_range,
                                 int n, float min_val, float max_val, float bin_width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = data[idx];
    int bin = (int)((val - min_val) / bin_width);

    if (bin >= 0 && bin < NUM_BINS)
    {
        atomicAdd(&bins[bin], 1);
    }
    else
    {
        atomicAdd(out_of_range, 1);
    }
}

int main()
{
    const int num_elements = 1 << 20; // 1M elements
    const float min_val = 0.0f;
    const float max_val = 10.0f;
    const float bin_width = (max_val - min_val) / NUM_BINS;

    // Allocate host memory
    float *h_data = (float *)malloc(num_elements * sizeof(float));
    if (!h_data)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Generate random data between -5 and 15 to include out-of-range values
    srand((unsigned)time(NULL));
    for (int i = 0; i < num_elements; ++i)
    {
        h_data[i] = -5.0f + ((float)rand() / RAND_MAX) * 20.0f; // range [-5, 15]
    }

    // Allocate device memory
    float *d_data;
    int *d_bins;
    int *d_out_of_range;

    cudaError_t err;

    err = cudaMalloc((void **)&d_data, num_elements * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_data failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    err = cudaMalloc((void **)&d_bins, NUM_BINS * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_bins failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return EXIT_FAILURE;
    }

    err = cudaMalloc((void **)&d_out_of_range, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_out_of_range failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        cudaFree(d_bins);
        return EXIT_FAILURE;
    }

    // Copy data to device
    err = cudaMemcpy(d_data, h_data, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy h_data->d_data failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        cudaFree(d_bins);
        cudaFree(d_out_of_range);
        return EXIT_FAILURE;
    }

    // Initialize bins and out_of_range on device to zero
    err = cudaMemset(d_bins, 0, NUM_BINS * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemset d_bins failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    err = cudaMemset(d_out_of_range, 0, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemset d_out_of_range failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_bins, d_out_of_range,
                                                        num_elements, min_val, max_val, bin_width);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    // Copy results back to host
    int h_bins[NUM_BINS];
    int h_out_of_range;

    err = cudaMemcpy(h_bins, d_bins, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_bins->h_bins failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMemcpy(&h_out_of_range, d_out_of_range, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_out_of_range->h_out_of_range failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    // Print histogram
    printf("Histogram (value range [%.2f, %.2f)):\n", min_val, max_val);
    for (int i = 0; i < NUM_BINS; ++i)
    {
        float bin_start = min_val + i * bin_width;
        float bin_end = bin_start + bin_width;
        printf("  Bin %2d [%.2f, %.2f): %d\n", i, bin_start, bin_end, h_bins[i]);
    }
    printf("Out-of-range count: %d\n", h_out_of_range);

cleanup:
    cudaFree(d_data);
    cudaFree(d_bins);
    cudaFree(d_out_of_range);
    free(h_data);
    return 0;
}
