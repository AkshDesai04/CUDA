```cpp
/*********************************************************************
 * Aim: Extend the pipeline to use 3 streams and 3 buffers. Does this
 * provide further speedup over 2 streams?
 *
 * Thinking:
 * -------------
 * The task is to illustrate how CUDA streams and multiple
 * double/triple‑buffering can overlap data transfer and kernel
 * execution, improving overall throughput.  In the classic 2‑stream
 * pipeline we have:
 *   Stream 0: copy a chunk from host to device, run kernel, copy back.
 *   Stream 1: same as stream 0 but started after the first chunk of
 *              stream 0 is transferred, so that while stream 0 is
 *              executing the kernel the host can start transferring
 *              the next chunk to the device.  This results in a
 *              pipeline of three stages that can overlap.
 *
 *  To extend this to 3 streams we add a third buffer on the device
 *  and another stream.  We will schedule transfers and kernel launches
 *  for chunks i in a round‑robin fashion over the three streams:
 *
 *    Stream 0: chunk 0, 3, 6, ...
 *    Stream 1: chunk 1, 4, 7, ...
 *    Stream 2: chunk 2, 5, 8, ...
 *
 *  Each stream owns its own pair of host‑side pinned memory buffers
 *  (for transfer in and out) and a device buffer.  This allows us to
 *  keep three chunks in flight concurrently.  The question is whether
 *  the additional stream and buffer provide any measurable speedup
 *  over the 2‑stream version, especially when transfer times are
 *  significant relative to kernel time.
 *
 *  Implementation details:
 *  - We use cudaMemcpyAsync for transfers, launching them on the
 *    appropriate stream.
 *  - Kernel launch also occurs on that stream, so the three stages
 *    (H→D copy, kernel, D→H copy) are serialized per stream but can
 *    overlap across streams.
 *  - We use cudaEvent_t to time each pipeline version separately.
 *  - To keep the example simple and self‑contained we operate on
 *    large vectors of floats, performing a simple addition kernel.
 *  - We compare the time taken for 2‑stream vs 3‑stream executions
 *    and print the speedup factor.
 *
 *  Note: The code is written in standard CUDA C++ and can be
 *  compiled with:
 *
 *      nvcc -o pipeline_test pipeline_test.cu
 *
 *  Running the program prints the timings and the speedup achieved
 *  by the 3‑stream pipeline over the 2‑stream pipeline.
 *********************************************************************/

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>

#define CHECK_CUDA(call)                                                   \
    {                                                                      \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    __FUNCTION__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }

// Simple elementwise addition kernel
__global__ void vecAdd(const float *a, const float *b, float *c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

// Helper to allocate pinned host memory for a chunk
float* allocPinned(size_t bytes) {
    float* ptr;
    CHECK_CUDA(cudaMallocHost((void**)&ptr, bytes));
    return ptr;
}

// Helper to free pinned host memory
void freePinned(float* ptr) {
    CHECK_CUDA(cudaFreeHost(ptr));
}

// Pipeline execution with given number of streams and buffers
float runPipeline(size_t totalSize, size_t chunkSize, int numStreams) {
    size_t elementsPerChunk = chunkSize;
    size_t bytesPerChunk = elementsPerChunk * sizeof(float);
    int numChunks = (totalSize + elementsPerChunk - 1) / elementsPerChunk;

    // Allocate device buffers
    float **d_bufs = (float**)malloc(numStreams * sizeof(float*));
    for (int s = 0; s < numStreams; ++s) {
        CHECK_CUDA(cudaMalloc((void**)&d_bufs[s], bytesPerChunk));
    }

    // Allocate host pinned buffers for input A and B, and output C
    float **h_a_bufs = (float**)malloc(numStreams * sizeof(float*));
    float **h_b_bufs = (float**)malloc(numStreams * sizeof(float*));
    float **h_c_bufs = (float**)malloc(numStreams * sizeof(float*));
    for (int s = 0; s < numStreams; ++s) {
        h_a_bufs[s] = allocPinned(bytesPerChunk);
        h_b_bufs[s] = allocPinned(bytesPerChunk);
        h_c_bufs[s] = allocPinned(bytesPerChunk);
    }

    // Allocate streams
    cudaStream_t *streams = (cudaStream_t*)malloc(numStreams * sizeof(cudaStream_t));
    for (int s = 0; s < numStreams; ++s) {
        CHECK_CUDA(cudaStreamCreate(&streams[s]));
    }

    // Events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Main loop over chunks
    for (int chunk = 0; chunk < numChunks; ++chunk) {
        int streamIdx = chunk % numStreams;
        size_t offset = chunk * elementsPerChunk;
        size_t currentChunkSize = (offset + elementsPerChunk <= totalSize) ?
                                  elementsPerChunk : (totalSize - offset);

        // Fill host input buffers with dummy data
        for (size_t i = 0; i < currentChunkSize; ++i) {
            h_a_bufs[streamIdx][i] = static_cast<float>(offset + i);
            h_b_bufs[streamIdx][i] = static_cast<float>(offset + i) * 2.0f;
        }

        // Async copy H->D for A and B
        CHECK_CUDA(cudaMemcpyAsync(d_bufs[streamIdx], h_a_bufs[streamIdx],
                                   currentChunkSize * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[streamIdx]));
        CHECK_CUDA(cudaMemcpyAsync(d_bufs[streamIdx] + elementsPerChunk, h_b_bufs[streamIdx],
                                   currentChunkSize * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[streamIdx]]);

        // Launch kernel on this stream
        int threadsPerBlock = 256;
        int blocksPerGrid = (currentChunkSize + threadsPerBlock - 1) / threadsPerBlock;
        vecAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[streamIdx]>>>(
            d_bufs[streamIdx],
            d_bufs[streamIdx] + elementsPerChunk,
            d_bufs[streamIdx] + 2 * elementsPerChunk,
            currentChunkSize);

        // Async copy D->H for C
        CHECK_CUDA(cudaMemcpyAsync(h_c_bufs[streamIdx], d_bufs[streamIdx] + 2 * elementsPerChunk,
                                   currentChunkSize * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[streamIdx]));
    }

    // Wait for all streams to finish
    for (int s = 0; s < numStreams; ++s) {
        CHECK_CUDA(cudaStreamSynchronize(streams[s]));
    }

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // Clean up
    for (int s = 0; s < numStreams; ++s) {
        CHECK_CUDA(cudaFree(d_bufs[s]));
        freePinned(h_a_bufs[s]);
        freePinned(h_b_bufs[s]);
        freePinned(h_c_bufs[s]);
        CHECK_CUDA(cudaStreamDestroy(streams[s]));
    }
    free(d_bufs);
    free(h_a_bufs);
    free(h_b_bufs);
    free(h_c_bufs);
    free(streams);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms;
}

int main() {
    size_t totalSize = 1 << 24;      // 16M elements (~64 MB)
    size_t chunkSize = 1 << 20;      // 1M elements (~4 MB)

    std::cout << "Running pipeline with 2 streams (double buffering)...\n";
    float time2 = runPipeline(totalSize, chunkSize, 2);
    std::cout << "Time (2 streams): " << time2 << " ms\n\n";

    std::cout << "Running pipeline with 3 streams (triple buffering)...\n";
    float time3 = runPipeline(totalSize, chunkSize, 3);
    std::cout << "Time (3 streams): " << time3 << " ms\n\n";

    if (time3 > 0) {
        float speedup = time2 / time3;
        std::cout << "Speedup (3 streams over 2 streams): " << std::fixed << std::setprecision(3)
                  << speedup << "x\n";
    }

    return 0;
}
```