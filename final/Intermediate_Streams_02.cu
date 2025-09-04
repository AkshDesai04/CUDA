```cuda
/*
Aim of the program:
Modify the previous program to use two CUDA streams. Overlap execution by issuing the work for chunk `k` and chunk `k+1` in a pipelined fashion: Copy(k, s1), Copy(k+1, s2), Exec(k, s1), Exec(k+1, s2), etc. Use `cudaStreamSynchronize` or CUDA events to manage dependencies. The host memory must be pinned.

Thinking and design notes:

1.  **Chunking strategy**  
    The problem requires processing a large vector (or array) in chunks.  
    Each chunk is processed independently by copying the relevant slice of the source data from host to device, executing a kernel that operates only on that slice, and then copying the result back to host.  
    A chunk is defined by a fixed size `CHUNK_SIZE`. The total number of chunks is `numChunks = (totalSize + CHUNK_SIZE - 1) / CHUNK_SIZE`.

2.  **Two CUDA streams and pipelining**  
    We use two streams (`stream[0]` and `stream[1]`).  
    The pipeline works as follows:
    - For chunk `k` (even index) we issue:
      - `cudaMemcpyAsync` of source data into device buffer belonging to stream 0
      - Kernel launch on stream 0
      - `cudaMemcpyAsync` result back to host on stream 0
    - For chunk `k+1` (odd index) we issue the same operations on stream 1.
    Because the copy for chunk `k+1` is issued on a different stream, it can execute concurrently with the kernel of chunk `k`.  
    The ordering guarantees are provided automatically by the stream semantics: within a stream, operations are serialized; between streams, they can run concurrently as long as resources allow.

3.  **Device buffers per stream**  
    Each stream uses its own set of device buffers (`d_a`, `d_b`, `d_c`) to avoid conflicts.  
    The buffer size is `CHUNK_SIZE * sizeof(float)`; the last chunk may be smaller, so the actual copy size is computed per iteration.

4.  **Pinned host memory**  
    Host arrays (`h_a`, `h_b`, `h_c`) are allocated with `cudaHostAlloc` to obtain pinned memory, which allows higher throughput for asynchronous copies.

5.  **Synchronization**  
    After all chunks are dispatched, we call `cudaStreamSynchronize` on each stream to ensure that all copies and kernel executions have finished before we read back or free resources.  
    We also use CUDA events (`start`, `stop`) to measure the total elapsed time of the pipelined execution.

6.  **Error checking**  
    A helper macro `CHECK_CUDA` is used to capture CUDA API errors and abort execution with a message.

7.  **Kernel**  
    The kernel performs a simple vector addition: `c[i] = a[i] + b[i]`.  
    The kernel launch parameters (`gridDim`, `blockDim`) are chosen based on the chunk size.

8.  **Verification**  
    After the execution, the program checks a few elements of the output array to confirm correctness.

9.  **Compilation**  
    The program is written in standard CUDA C and can be compiled with:
    `nvcc -o pipelined_streams pipelined_streams.cu`

This design satisfies all requirements: two streams, pipelined copy/exec, pinned host memory, and explicit synchronization. It demonstrates how to overlap data transfer and computation effectively on a GPU.
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",            \
                    __FILE__, __LINE__, err, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Simple vector addition kernel
__global__ void addKernel(const float* a, const float* b, float* c, int offset, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    // Parameters
    const size_t totalSize = 1 << 20;      // 1,048,576 elements (~4 MB)
    const size_t chunkSize = 1 << 15;      // 32,768 elements (~128 KB)
    const int numChunks = (totalSize + chunkSize - 1) / chunkSize;

    printf("Total size: %zu elements, Chunk size: %zu, Num chunks: %d\n",
           totalSize, chunkSize, numChunks);

    // Allocate pinned host memory
    float *h_a, *h_b, *h_c;
    CHECK_CUDA(cudaHostAlloc((void**)&h_a, totalSize * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc((void**)&h_b, totalSize * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc((void**)&h_c, totalSize * sizeof(float), cudaHostAllocDefault));

    // Initialize input data
    for (size_t i = 0; i < totalSize; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Create two streams
    cudaStream_t streams[2];
    CHECK_CUDA(cudaStreamCreate(&streams[0]));
    CHECK_CUDA(cudaStreamCreate(&streams[1]));

    // Device buffers per stream
    float *d_a[2], *d_b[2], *d_c[2];
    for (int s = 0; s < 2; ++s) {
        CHECK_CUDA(cudaMalloc((void**)&d_a[s], chunkSize * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&d_b[s], chunkSize * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&d_c[s], chunkSize * sizeof(float)));
    }

    // Timing events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, 0));

    // Launch pipeline
    for (int chunk = 0; chunk < numChunks; ++chunk) {
        int s = chunk % 2; // stream index
        size_t offset = (size_t)chunk * chunkSize;
        size_t curChunkSize = (offset + chunkSize <= totalSize) ?
                               chunkSize : totalSize - offset;

        // Copy input data to device
        CHECK_CUDA(cudaMemcpyAsync(d_a[s], h_a + offset,
                                   curChunkSize * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[s]));
        CHECK_CUDA(cudaMemcpyAsync(d_b[s], h_b + offset,
                                   curChunkSize * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[s]]);

        // Kernel launch
        int threadsPerBlock = 256;
        int blocksPerGrid = (curChunkSize + threadsPerBlock - 1) / threadsPerBlock;
        addKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[s]]>(d_a[s], d_b[s], d_c[s], 0, curChunkSize);

        // Copy result back to host
        CHECK_CUDA(cudaMemcpyAsync(h_c + offset, d_c[s],
                                   curChunkSize * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[s]));
    }

    // Wait for all streams to finish
    CHECK_CUDA(cudaStreamSynchronize(streams[0]));
    CHECK_CUDA(cudaStreamSynchronize(streams[1]));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Total elapsed time: %f ms\n", milliseconds);

    // Verify results (first 10 elements)
    int errors = 0;
    for (int i = 0; i < 10; ++i) {
        if (h_c[i] != 3.0f) {
            printf("Error at index %d: %f (expected 3.0)\n", i, h_c[i]);
            errors++;
        }
    }
    if (errors == 0) {
        printf("Verification passed!\n");
    } else {
        printf("Verification failed with %d errors.\n", errors);
    }

    // Clean up
    for (int s = 0; s < 2; ++s) {
        CHECK_CUDA(cudaFree(d_a[s]));
        CHECK_CUDA(cudaFree(d_b[s]));
        CHECK_CUDA(cudaFree(d_c[s]));
    }
    CHECK_CUDA(cudaStreamDestroy(streams[0]));
    CHECK_CUDA(cudaStreamDestroy(streams[1]));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
```