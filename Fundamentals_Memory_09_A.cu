```cpp
/*
Aim of the program: 
Explain in a comment why pinned memory transfers are faster. (Hint: DMA, no intermediate staging buffer).

Thinking: 
Pinned (page-locked) host memory is used in CUDA to improve transfer performance between host and device.
When host memory is pinned, the operating system guarantees that the memory pages remain resident in physical RAM and are not swapped out. 
This allows the GPU's Direct Memory Access (DMA) engine to directly read from or write to these pages without having to copy data through an intermediate staging buffer in pageable memory. 
In pageable memory, CUDA first copies data to a staging buffer that is page-locked, performs the DMA transfer, and then copies back from the staging buffer, adding extra memory copies and CPU overhead. 
Pinned memory eliminates this extra step, reduces latency, and can increase throughput because the DMA engine can stream data directly from the host to the device or vice versa. 
Additionally, pinned memory can be used with asynchronous transfers and streams, allowing overlap of communication and computation, further improving overall performance. 
*/

#include <cuda_runtime.h>
#include <iostream>

__global__ void dummyKernel(float *d_data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_data[idx] = d_data[idx] * 2.0f;
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate pinned host memory
    float *h_data;
    cudaMallocHost(&h_data, size); // pinned memory

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_data;
    cudaMalloc(&d_data, size);

    // Create CUDA stream for asynchronous transfer
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Asynchronously copy from host (pinned) to device
    cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dummyKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, N);

    // Asynchronously copy result back to host
    cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream);

    // Wait for all operations in the stream to complete
    cudaStreamSynchronize(stream);

    // Verify a few results
    bool success = true;
    for (int i = 0; i < 5; ++i) {
        if (h_data[i] != static_cast<float>(i) * 2.0f) {
            success = false;
            break;
        }
    }
    std::cout << (success ? "Success" : "Failure") << std::endl;

    // Clean up
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    cudaFreeHost(h_data);

    return 0;
}
```