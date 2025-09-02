```cpp
/* 
Aim of the program:
Plot the effective bandwidth (GB/s) of both kernels for various matrix sizes.

Thinking:
The goal is to measure how two different CUDA kernels perform in terms of memory
bandwidth when copying a square matrix of floating point numbers.  The first kernel
(`naiveCopy`) performs a simple global memory copy with one thread per element.
The second kernel (`tiledCopy`) uses a 2‑D block and shared memory to bring the
matrix in tiles, which often improves coalescing and reduces memory traffic.

For each matrix size (from 512×512 up to 8192×8192, stepping by 512), we:
1. Allocate device memory for the source and destination matrices.
2. Initialize the source matrix with some dummy data.
3. Launch each kernel, timing the execution with CUDA events.
4. Compute the effective bandwidth as:
        bandwidth = (2 * N * N * sizeof(float)) / elapsed_time_seconds
   (two passes of N×N elements: one read and one write).
5. Output the size and bandwidths in a CSV file `bandwidth.csv` so that the user
   can plot the results with any external plotting tool (e.g., gnuplot or Python).

The program also includes a simple CUDA error‑checking macro and uses
`cudaEvent_t` for high‑resolution timing.  All code is self‑contained in a
single .cu file. 
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Kernel 1: Naive global memory copy
__global__ void naiveCopy(const float* src, float* dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = N * N;
    for (int i = idx; i < total; i += stride) {
        dst[i] = src[i];
    }
}

// Kernel 2: Tiled copy using shared memory (16x16 tiles)
__global__ void tiledCopy(const float* src, float* dst, int N) {
    __shared__ float tile[16][17];  // +1 to avoid bank conflicts
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;

    // Load into shared memory
    if (row < N && col < N) {
        int idx = row * N + col;
        tile[ty][tx] = src[idx];
    }
    __syncthreads();

    // Write back from shared memory
    if (row < N && col < N) {
        int idx = row * N + col;
        dst[idx] = tile[ty][tx];
    }
}

// Helper function to run a kernel, time it, and compute bandwidth
float runKernelAndMeasure(void (*kernel)(const float*, float*, int), 
                          const float* d_src, float* d_dst, int N,
                          dim3 gridDim, dim3 blockDim) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    // Launch the kernel
    kernel<<<gridDim, blockDim>>>(d_src, d_dst, N);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Clean up events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Bandwidth in GB/s: 2 * N*N * sizeof(float) bytes over ms milliseconds
    double bytes = 2.0 * N * N * sizeof(float);
    double seconds = ms / 1000.0;
    double gbps = bytes / (seconds * 1e9);
    return static_cast<float>(gbps);
}

int main() {
    // Open output CSV file
    FILE* fp = fopen("bandwidth.csv", "w");
    if (!fp) {
        fprintf(stderr, "Failed to open output file.\n");
        return EXIT_FAILURE;
    }
    fprintf(fp, "Size,Naive_GBps,Tiled_GBps\n");

    // Loop over matrix sizes
    for (int N = 512; N <= 8192; N *= 2) {  // Sizes: 512, 1024, 2048, 4096, 8192
        size_t bytes = N * N * sizeof(float);
        // Allocate device memory
        float *d_src, *d_dst;
        CUDA_CHECK(cudaMalloc((void**)&d_src, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_dst, bytes));

        // Initialize source matrix on host
        float* h_src = (float*)malloc(bytes);
        for (size_t i = 0; i < N * N; ++i) {
            h_src[i] = static_cast<float>(i) * 0.001f;
        }
        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));
        free(h_src);

        // Define grid/block dimensions
        dim3 blockDimNaive(256);
        dim3 gridDimNaive((N * N + blockDimNaive.x - 1) / blockDimNaive.x);
        dim3 blockDimTiled(16, 16);
        dim3 gridDimTiled((N + 15) / 16, (N + 15) / 16);

        // Measure naive kernel
        float bandwidthNaive = runKernelAndMeasure(naiveCopy, d_src, d_dst, N,
                                                   gridDimNaive, blockDimNaive);

        // Measure tiled kernel
        float bandwidthTiled = runKernelAndMeasure(tiledCopy, d_src, d_dst, N,
                                                   gridDimTiled, blockDimTiled);

        // Print to console
        printf("Size %4d: Naive %.3f GB/s, Tiled %.3f GB/s\n",
               N, bandwidthNaive, bandwidthTiled);

        // Write to CSV
        fprintf(fp, "%d,%.3f,%.3f\n", N, bandwidthNaive, bandwidthTiled);

        // Clean up device memory
        CUDA_CHECK(cudaFree(d_src));
        CUDA_CHECK(cudaFree(d_dst));
    }

    fclose(fp);
    printf("Bandwidth data written to bandwidth.csv\n");
    return 0;
}
```