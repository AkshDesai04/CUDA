```cuda
/*
Aim of the program: Use Nsight Compute to visualize the memory chart and see the inefficiency of the strided access.

Thinking:
To demonstrate the inefficiency of strided memory accesses we implement two simple copy kernels.
1. coalescedKernel – copies src to dst directly (stride = 1). Each thread reads a contiguous memory word and writes it back, giving optimal coalescing.
2. stridedKernel – each thread reads src at an offset of `stride` (e.g., 32). This creates a pattern where consecutive threads in a warp access memory locations that are 32 * sizeof(float) bytes apart. This pattern is non‑coalesced and forces the GPU to perform many more memory transactions, increasing latency and reducing bandwidth utilization.

The program allocates a large array, fills it with random data, and then runs both kernels while measuring elapsed time using CUDA events. After compilation, the user can launch the binary with Nsight Compute (e.g., `ncu --set full -o output.ncu-rep ./stride`) and open the report to inspect the memory chart. In the report, the strided kernel should show a higher number of global memory transactions and lower memory throughput compared to the coalesced kernel, illustrating the inefficiency caused by the stride.

The key parameters:
- N: number of elements processed per kernel (size of dst).
- stride: distance between consecutive reads in the src array.
- blockSize: number of threads per block (256 is chosen for simplicity).

By comparing the execution times and the memory charts, the user will observe the difference in memory access patterns.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK(call)                                                     \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Coalesced copy kernel (stride = 1)
__global__ void coalescedKernel(float *dst, const float *src, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        dst[idx] = src[idx];
}

// Strided copy kernel (stride > 1)
__global__ void stridedKernel(float *dst, const float *src, int stride, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        dst[idx] = src[idx * stride];
}

// Utility to fill an array with random floats
void fillRandom(float *arr, size_t size)
{
    for (size_t i = 0; i < size; ++i)
        arr[i] = static_cast<float>(rand()) / RAND_MAX;
}

int main(void)
{
    const int blockSize = 256;
    const int N = 1 << 18;          // Number of elements in dst
    const int stride = 32;          // Stride for stridedKernel
    const size_t srcSize = static_cast<size_t>(N) * stride;
    const size_t dstSize = N;

    // Allocate host memory
    float *h_src = (float *)malloc(srcSize * sizeof(float));
    float *h_dst = (float *)malloc(dstSize * sizeof(float));
    if (!h_src || !h_dst) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host source array
    srand(0);
    fillRandom(h_src, srcSize);

    // Allocate device memory
    float *d_src, *d_dst;
    CHECK(cudaMalloc((void **)&d_src, srcSize * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_dst, dstSize * sizeof(float)));

    // Copy data from host to device
    CHECK(cudaMemcpy(d_src, h_src, srcSize * sizeof(float), cudaMemcpyHostToDevice));

    // Determine grid size
    int gridSize = (N + blockSize - 1) / blockSize;

    // Timing events
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // ------------------- Coalesced kernel -------------------
    CHECK(cudaEventRecord(start));
    coalescedKernel<<<gridSize, blockSize>>>(d_dst, d_src, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float coalescedTime = 0;
    CHECK(cudaEventElapsedTime(&coalescedTime, start, stop));
    printf("Coalesced kernel elapsed time: %.3f ms\n", coalescedTime);

    // ------------------- Strided kernel --------------------
    CHECK(cudaEventRecord(start));
    stridedKernel<<<gridSize, blockSize>>>(d_dst, d_src, stride, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float stridedTime = 0;
    CHECK(cudaEventElapsedTime(&stridedTime, start, stop));
    printf("Strided kernel elapsed time:   %.3f ms\n", stridedTime);

    // Copy result back to host (not used further but ensures kernel finished)
    CHECK(cudaMemcpy(h_dst, d_dst, dstSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Clean up
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    free(h_src);
    free(h_dst);

    return 0;
}
```