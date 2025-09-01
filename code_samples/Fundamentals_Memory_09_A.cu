/*
Explain in a comment why pinned memory transfers are faster. (Hint: DMA, no intermediate staging buffer).

My thinking:
- In CUDA, data transfers between host (CPU) and device (GPU) are performed by the GPU’s Direct Memory Access (DMA) engine.
- If the host memory is pageable (regular malloc), the CUDA driver must first copy that data to an intermediate, internally‑allocated pinned buffer before the DMA engine can read it. This double‑copy adds overhead and latency.
- Pinned (page‑locked) memory, allocated via cudaHostAlloc or cudaMallocHost, cannot be paged out by the operating system. The driver can map it directly into the GPU’s address space, allowing the DMA engine to transfer data in one step, eliminating the intermediate staging buffer.
- The result is lower transfer latency and higher throughput, especially for large or frequent transfers. Additionally, pinned memory can enable asynchronous transfers overlapping with kernel execution.

The program below demonstrates the difference by timing a large memcpy from pageable to device memory versus from pinned to device memory. The pinned transfer is noticeably faster, illustrating the benefit of DMA without the intermediate buffer.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dummyKernel(int *d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_data[idx] += 1; // simple operation to keep GPU busy
    }
}

int main(void) {
    const size_t N = 1 << 24; // 16M elements (~64 MB)
    const size_t bytes = N * sizeof(int);

    // Allocate pageable host memory
    int *h_pageable = (int*)malloc(bytes);
    if (!h_pageable) {
        fprintf(stderr, "Failed to allocate pageable host memory\n");
        return -1;
    }
    // Initialize
    for (size_t i = 0; i < N; ++i) h_pageable[i] = i;

    // Allocate pinned host memory
    int *h_pinned = NULL;
    cudaError_t err = cudaHostAlloc((void**)&h_pinned, bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaHostAlloc failed: %s\n", cudaGetErrorString(err));
        free(h_pageable);
        return -1;
    }
    // Initialize pinned memory
    for (size_t i = 0; i < N; ++i) h_pinned[i] = i;

    // Allocate device memory
    int *d_data = NULL;
    err = cudaMalloc((void**)&d_data, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_pageable);
        cudaFreeHost(h_pinned);
        return -1;
    }

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 1) Pageable host -> device
    cudaEventRecord(start, 0);
    err = cudaMemcpy(d_data, h_pageable, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float pageable_ms = 0;
    cudaEventElapsedTime(&pageable_ms, start, stop);
    printf("Pageable memory transfer time: %.3f ms\n", pageable_ms);

    // 2) Pinned host -> device
    cudaEventRecord(start, 0);
    err = cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float pinned_ms = 0;
    cudaEventElapsedTime(&pinned_ms, start, stop);
    printf("Pinned memory transfer time: %.3f ms\n", pinned_ms);

    // Launch dummy kernel to use the data
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dummyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    cudaDeviceSynchronize();

    // Clean up
    cudaFree(d_data);
    cudaFreeHost(h_pinned);
    free(h_pageable);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
