/*
Explain in a comment why pinned memory transfers are faster. (Hint: DMA, no intermediate staging buffer).

Pinned (page-locked) memory allows the GPU to perform direct memory access (DMA) between host and device without an intermediate staging buffer. When memory is not pinned, the CUDA driver must first copy the data into a temporary pinned buffer before initiating the transfer. This additional copy step introduces extra latency and consumes CPU cycles. With pinned memory, the data can be read directly by the DMA engine, resulting in faster transfer rates and lower overhead. Additionally, pinned memory ensures that the physical addresses of the pages remain stable, which simplifies the DMA engine's operation and reduces fragmentation-related delays. Overall, by eliminating the staging buffer and enabling direct DMA, pinned memory transfers achieve higher throughput and lower latency compared to pageable memory transfers.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    // Example: Allocate pinned memory (optional)
    size_t size = 1024 * 1024; // 1 MB
    void* hostPtr;
    cudaMallocHost(&hostPtr, size); // pinned allocation

    // Allocate device memory
    void* devicePtr;
    cudaMalloc(&devicePtr, size);

    // Simple host-to-device copy
    cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice);

    // Clean up
    cudaFree(devicePtr);
    cudaFreeHost(hostPtr);

    printf("Pinned memory transfer example completed.\n");
    return 0;
}
