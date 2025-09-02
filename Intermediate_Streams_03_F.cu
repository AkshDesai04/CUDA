```cuda
/*
Aim of the program:
Plot the total execution time vs. the number of streams used (1, 2, 3, 4...). The benefit usually saturates quickly.

Thinking:
1. We need to demonstrate how using multiple CUDA streams can overlap work and potentially reduce total execution time.
2. The simplest way to measure this is to launch a small, compute‑intensive kernel multiple times, once per stream.
   Each kernel will operate on a distinct portion of a large array to avoid any data hazards.
3. For each number of streams S (from 1 up to a chosen maximum), we:
   a) Create S streams.
   b) Allocate a contiguous device array of size S * N (where N is the number of elements each kernel will process).
   c) For each stream i, launch the kernel on the slice starting at offset i * N.
   d) Time the total execution from just before launching the first kernel to just after all kernels have completed.
4. Timing is performed using CUDA events. Because we use the legacy default stream (stream 0) for recording the start and end events,
   stream 0 implicitly synchronizes with all other streams. Thus the end event will only be recorded after all kernels in all streams
   have finished executing, giving us the true concurrent total time.
5. After each experiment we print the number of streams and the elapsed time in milliseconds. The user can then plot this output
   externally (e.g., using gnuplot, Python matplotlib, Excel, etc.) to observe how the benefit of additional streams saturates.
6. Error checking is performed after CUDA API calls to catch any issues early.
7. The kernel is intentionally simple (incrementing array elements) to keep the overhead small and focus on stream concurrency.
8. We keep the problem size moderate (N = 1<<20 ≈ 1M elements per stream, 4 MB per stream) so that memory usage stays reasonable
   while still being large enough to potentially benefit from concurrency.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Simple kernel that increments each element by 1
__global__ void inc_kernel(int *data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        data[idx] += 1;
}

int main()
{
    // Problem parameters
    const int N = 1 << 20;          // Elements per stream (≈1M)
    const int maxStreams = 8;       // Maximum number of streams to test
    const int threadsPerBlock = 256;
    const int blocksPerGrid  = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Number of Streams | Elapsed Time (ms)\n");
    printf("-------------------------------------\n");

    for (int streamCount = 1; streamCount <= maxStreams; ++streamCount)
    {
        // Allocate device memory for all streams
        size_t totalSize = streamCount * N * sizeof(int);
        int *d_data = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_data, totalSize));

        // Create streams
        cudaStream_t *streams = new cudaStream_t[streamCount];
        for (int i = 0; i < streamCount; ++i)
        {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }

        // Create events for timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Record start event in legacy default stream (stream 0)
        CUDA_CHECK(cudaEventRecord(start, 0));

        // Launch kernel on each stream
        for (int i = 0; i < streamCount; ++i)
        {
            int *streamPtr = d_data + i * N; // Offset for this stream
            inc_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(streamPtr, N);
            // Check for launch errors
            CUDA_CHECK(cudaGetLastError());
        }

        // Record stop event in legacy default stream (will wait for all streams)
        CUDA_CHECK(cudaEventRecord(stop, 0));

        // Wait for event to complete
        CUDA_CHECK(cudaEventSynchronize(stop));

        // Compute elapsed time
        float elapsedMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));

        // Output result
        printf("%17d | %18.3f\n", streamCount, elapsedMs);

        // Cleanup
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        for (int i = 0; i < streamCount; ++i)
        {
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }
        delete[] streams;
        CUDA_CHECK(cudaFree(d_data));
    }

    return 0;
}
```