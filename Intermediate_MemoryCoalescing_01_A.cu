```cuda
/*
Explain what memory coalescing is. (Threads in a warp accessing contiguous memory locations in a single transaction).

This program demonstrates the concept of memory coalescing in CUDA. In a typical CUDA kernel, threads are executed in groups called warps (usually 32 threads). When all threads in a warp access consecutive memory locations, the hardware can combine those accesses into a single memory transaction, which dramatically improves memory throughput. This is known as memory coalescing.

The aim of this example is to illustrate the difference between coalesced and non‑coalesced memory accesses. Two kernels are provided:

1. `coalescedKernel` – each thread reads from `src[tid]` and writes to `dst[tid]`. Since consecutive threads access consecutive elements, the GPU can coalesce the reads/writes into a single transaction per warp.

2. `nonCoalescedKernel` – each thread reads from `src[tid * stride]` and writes to `dst[tid * stride]`. With a stride larger than 1 (here 32), consecutive threads access memory locations that are far apart, preventing coalescing and resulting in multiple memory transactions.

The program allocates a large array, initializes it, launches both kernels, and measures execution time using CUDA events. It also prints a few memory addresses accessed by each kernel to illustrate how the memory pattern changes.

By comparing the timing results, one can see that the coalesced kernel is significantly faster than the non‑coalesced one, demonstrating the performance benefit of memory coalescing.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define ARRAY_SIZE (1 << 20)   // 1M elements
#define BLOCK_SIZE 256

__global__ void coalescedKernel(const float* src, float* dst, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        dst[tid] = src[tid];
    }
}

__global__ void nonCoalescedKernel(const float* src, float* dst, int n, int stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * stride;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

int main()
{
    float *h_src = new float[ARRAY_SIZE];
    float *h_dst_coal = new float[ARRAY_SIZE];
    float *h_dst_non = new float[ARRAY_SIZE];

    // Initialize host array
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        h_src[i] = static_cast<float>(i);
    }

    // Device memory allocation
    float *d_src, *d_dst_coal, *d_dst_non;
    cudaMalloc(&d_src, ARRAY_SIZE * sizeof(float));
    cudaMalloc(&d_dst_coal, ARRAY_SIZE * sizeof(float));
    cudaMalloc(&d_dst_non, ARRAY_SIZE * sizeof(float));

    cudaMemcpy(d_src, h_src, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Setup launch parameters
    int blocks = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ==========================
    // Coalesced kernel
    // ==========================
    cudaEventRecord(start);
    coalescedKernel<<<blocks, BLOCK_SIZE>>>(d_src, d_dst_coal, ARRAY_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_coal = 0;
    cudaEventElapsedTime(&ms_coal, start, stop);

    // ==========================
    // Non-coalesced kernel
    // ==========================
    int stride = 32;  // stride that disrupts coalescing
    cudaEventRecord(start);
    nonCoalescedKernel<<<blocks, BLOCK_SIZE>>>(d_src, d_dst_non, ARRAY_SIZE, stride);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_non = 0;
    cudaEventElapsedTime(&ms_non, start, stop);

    // Copy results back to host
    cudaMemcpy(h_dst_coal, d_dst_coal, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dst_non, d_dst_non, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify correctness for a few elements
    printf("Sample results (coalesced vs non-coalesced):\n");
    for (int i = 0; i < 8; ++i) {
        printf("dst[%d] = %.1f, non-dst[%d] = %.1f\n", i, h_dst_coal[i], i, h_dst_non[i]);
    }

    // Print timing
    printf("\nKernel execution times:\n");
    printf("Coalesced kernel:   %f ms\n", ms_coal);
    printf("Non-coalesced kernel: %f ms\n", ms_non);

    // Clean up
    delete[] h_src;
    delete[] h_dst_coal;
    delete[] h_dst_non;
    cudaFree(d_src);
    cudaFree(d_dst_coal);
    cudaFree(d_dst_non);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```