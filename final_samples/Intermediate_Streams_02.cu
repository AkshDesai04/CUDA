```cpp
/* 
Aim of the program: Modify the previous program to use two CUDA streams. Overlap execution by issuing the work for chunk k and chunk k+1 in a pipelined fashion: Copy(k, s1), Copy(k+1, s2), Exec(k, s1), Exec(k+1, s2), etc. Use cudaStreamSynchronize or CUDA events to manage dependencies. The host memory must be pinned.
  
Thinking process:
1. The program performs a simple vector addition: c = a + b. The data is processed in chunks to demonstrate streaming.
2. Two CUDA streams (s0 and s1) are created. For each chunk, the copy (host→device) and kernel execution are performed in the same stream. Because we alternate streams for consecutive chunks, a copy for chunk k+1 can overlap with the kernel execution of chunk k.
3. Pinned (page‑locked) host memory is allocated using cudaMallocHost so that asynchronous copies can be used.
4. Device memory is allocated for the entire arrays; kernel accesses elements using an offset to handle each chunk.
5. After all chunks are dispatched, we synchronize both streams and copy the final result back to the host.
6. Basic error checking is provided via the CUDA_CHECK macro.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// Error checking macro
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            std::cerr << "CUDA error at " << __FILE__ << ":"       \
                      << __LINE__ << " code=" << err                \
                      << " (" << cudaGetErrorString(err) << ")\n";  \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

// Kernel for vector addition
__global__ void vectorAddKernel(const float *a, const float *b, float *c,
                                size_t offset, size_t chunkSize)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t globalIdx = offset + idx;
    if (globalIdx < offset + chunkSize) {
        c[globalIdx] = a[globalIdx] + b[globalIdx];
    }
}

int main()
{
    const size_t N = 1 << 24;          // Total number of elements (~16M)
    const size_t chunkSize = 1 << 20;  // Chunk size (~1M elements)
    const size_t numChunks = (N + chunkSize - 1) / chunkSize;

    // Allocate pinned host memory
    float *h_a, *h_b, *h_c;
    CUDA_CHECK(cudaMallocHost(&h_a, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_b, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_c, N * sizeof(float)));

    // Initialize host data
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    // Create two CUDA streams
    cudaStream_t streams[2];
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));

    const size_t threadsPerBlock = 256;
    const size_t blocksPerGrid  = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;

    // Process each chunk
    for (size_t k = 0; k < numChunks; ++k) {
        cudaStream_t stream = streams[k % 2];
        size_t offset = k * chunkSize;
        size_t curChunkSize = (k == numChunks - 1) ? (N - offset) : chunkSize;

        // Asynchronously copy this chunk to device
        CUDA_CHECK(cudaMemcpyAsync(d_a + offset, h_a + offset,
                                   curChunkSize * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_b + offset, h_b + offset,
                                   curChunkSize * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));

        // Launch kernel for this chunk
        vectorAddKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>
            (d_a, d_b, d_c, offset, curChunkSize);

        // Optional: check kernel launch error
        CUDA_CHECK(cudaGetLastError());
    }

    // Wait for all streams to finish
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(streams[1]));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    for (size_t i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at index " << i
                      << ": got " << h_c[i] << ", expected " << expected << "\n";
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "All results correct.\n";
    }

    // Cleanup
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaStreamDestroy(streams[0]));
    CUDA_CHECK(cudaStreamDestroy(streams[1]));

    return 0;
}
```