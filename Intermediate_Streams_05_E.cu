```cuda
/*
Use a callback in your multi-chunk processing loop. When a chunk's DtoH copy is finished, the callback can trigger the host-side processing of that chunk's data, further improving pipelining.

Thinking:
- The goal is to demonstrate how CUDA streams and host callbacks can be combined to create a pipelined data processing workflow.
- We split a large array into fixed-size chunks so that each chunk can be processed independently.
- For each chunk we:
  1. Asynchronously copy the chunk from host to device.
  2. Launch a simple kernel to perform a per-element operation (increment).
  3. Asynchronously copy the processed chunk back to host.
- After the DtoH copy completes we register a host callback using `cudaLaunchHostFunc`. This callback runs on the host side and is guaranteed to execute after all preceding operations on the stream are finished.
- Inside the callback we perform some host-side work – here we compute the sum of the processed chunk and print a small diagnostic message. This demonstrates that host work can be overlapped with GPU work, as the callback does not block the stream.
- We use pinned host memory (`cudaHostAlloc`) so that the asynchronous copies can actually be overlapped.
- Error checking is performed after each CUDA call. We keep the example small and focused on the streaming and callback mechanism rather than on large data or complex kernels.
- At the end we synchronize the device, free all resources, and exit.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Size of the whole array and chunk size
#define TOTAL_ELEMENTS  (1 << 20)   // 1M elements
#define CHUNK_SIZE      (1 << 14)   // 16K elements
#define NUM_STREAMS     4

// Simple kernel that increments each element by 1
__global__ void incKernel(int *d_data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        d_data[idx] += 1;
}

// Structure to pass information to the host callback
struct ChunkInfo {
    int     chunkIdx;   // Index of the chunk
    int    *h_ptr;      // Host pointer to the chunk data
    size_t  size;       // Size of the chunk in elements
};

// Host callback function called after the stream finishes
void CUDART_CB hostCallback(cudaStream_t stream, cudaError_t status, void *userData)
{
    if (status != cudaSuccess) {
        fprintf(stderr, "Callback error: %s\n", cudaGetErrorString(status));
        return;
    }

    ChunkInfo *info = (ChunkInfo *)userData;
    long long sum = 0;
    for (size_t i = 0; i < info->size; ++i) {
        sum += info->h_ptr[i];
    }
    printf("Chunk %d processed. Sum = %lld\n", info->chunkIdx, sum);
}

int main(void)
{
    // Allocate pinned host memory for the whole array
    int *h_data;
    size_t bytes = TOTAL_ELEMENTS * sizeof(int);
    cudaError_t err = cudaHostAlloc((void **)&h_data, bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaHostAlloc failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (int i = 0; i < TOTAL_ELEMENTS; ++i) h_data[i] = i;

    // Allocate device memory
    int *d_data;
    err = cudaMalloc((void **)&d_data, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_data);
        return EXIT_FAILURE;
    }

    // Create multiple streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        err = cudaStreamCreate(&streams[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
            // Clean up previously created streams
            for (int j = 0; j < i; ++j) cudaStreamDestroy(streams[j]);
            cudaFree(d_data);
            cudaFreeHost(h_data);
            return EXIT_FAILURE;
        }
    }

    // Number of chunks
    int numChunks = (TOTAL_ELEMENTS + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Allocate array of ChunkInfo structures for callbacks
    ChunkInfo *chunkInfos = (ChunkInfo *)malloc(numChunks * sizeof(ChunkInfo));
    if (!chunkInfos) {
        fprintf(stderr, "malloc failed\n");
        for (int i = 0; i < NUM_STREAMS; ++i) cudaStreamDestroy(streams[i]);
        cudaFree(d_data);
        cudaFreeHost(h_data);
        return EXIT_FAILURE;
    }

    // Launch the multi-chunk processing loop
    for (int c = 0; c < numChunks; ++c) {
        int streamIdx = c % NUM_STREAMS;
        int offset = c * CHUNK_SIZE;
        int chunkSize = (offset + CHUNK_SIZE <= TOTAL_ELEMENTS) ? CHUNK_SIZE : TOTAL_ELEMENTS - offset;
        size_t chunkBytes = chunkSize * sizeof(int);

        // Set up chunk info for callback
        chunkInfos[c].chunkIdx = c;
        chunkInfos[c].h_ptr = h_data + offset;
        chunkInfos[c].size   = chunkSize;

        // Asynchronously copy chunk to device
        err = cudaMemcpyAsync(d_data + offset, h_data + offset, chunkBytes,
                              cudaMemcpyHostToDevice, streams[streamIdx]);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyAsync HtoD failed: %s\n", cudaGetErrorString(err));
            break;
        }

        // Launch kernel on this chunk
        int threadsPerBlock = 256;
        int blocksPerGrid = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;
        incKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[streamIdx]>>>(d_data + offset, chunkSize);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
            break;
        }

        // Asynchronously copy result back to host
        err = cudaMemcpyAsync(h_data + offset, d_data + offset, chunkBytes,
                              cudaMemcpyDeviceToHost, streams[streamIdx]);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyAsync DtoH failed: %s\n", cudaGetErrorString(err));
            break;
        }

        // Register host callback to process this chunk after copy completes
        err = cudaLaunchHostFunc(streams[streamIdx], hostCallback, &chunkInfos[c]);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaLaunchHostFunc failed: %s\n", cudaGetErrorString(err));
            break;
        }
    }

    // Wait for all streams to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
    }

    // Clean up
    for (int i = 0; i < NUM_STREAMS; ++i) cudaStreamDestroy(streams[i]);
    free(chunkInfos);
    cudaFree(d_data);
    cudaFreeHost(h_data);

    return EXIT_SUCCESS;
}
```