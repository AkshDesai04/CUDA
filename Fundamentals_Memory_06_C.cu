/*
Explain in a comment why memory leaks are bad, especially on a shared resource like a GPU.

Memory leaks occur when a program allocates memory but fails to release it back to the system. On a GPU, which is a highly shared and finite resource, this has several serious consequences:

1. **Finite device memory** – GPUs typically have limited memory (e.g., 8 GB, 16 GB, or even 24 GB for high‑end cards). A leak consumes part of this pool permanently, reducing the amount available for other kernels, buffers, or concurrent applications.

2. **Resource exhaustion** – If the leaked memory accumulates across many kernel launches or over time, the device may run out of memory, causing subsequent allocations or kernel launches to fail. This can lead to crashes or the need to restart the GPU.

3. **Performance degradation** – Even if memory is not fully exhausted, fragmenting the memory heap can increase allocation overhead and lead to suboptimal caching or higher copy latencies, reducing overall performance.

4. **Shared environment impact** – In multi‑tenant or multi‑process environments (e.g., GPU servers, cloud GPUs), a leaking application can starve other users of GPU resources, causing fairness and SLA violations.

5. **Debugging difficulty** – GPU memory leaks are harder to detect than CPU leaks because diagnostic tools often run asynchronously. The leak may not surface until the application has been running for a long time or under heavy load.

Therefore, proper allocation and deallocation (or using RAII wrappers) are essential to maintain GPU stability, performance, and fairness in shared systems. 
*/

#include <cuda_runtime.h>
#include <iostream>

// Dummy kernel to keep the file non‑empty
__global__ void dummyKernel() {}

// Main function that does nothing but demonstrates allocation and deallocation
int main() {
    // Allocate some GPU memory
    float* d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));

    // Launch the dummy kernel
    dummyKernel<<<1, 1>>>();

    // Free the allocated memory
    cudaFree(d_data);

    std::cout << "Memory allocation and deallocation performed. "
              << "No leaks introduced.\n";
    return 0;
}