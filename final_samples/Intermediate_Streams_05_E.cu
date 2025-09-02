```cuda
/*
Aim of the program:
Use a callback in your multi-chunk processing loop. When a chunk's DtoH copy is finished, the callback can trigger the host-side processing of that chunk's data, further improving pipelining.

Thinking:
1. Allocate a large array on the device and a corresponding host buffer.
2. Divide the data into fixed-size chunks and process them in a loop.
3. For each chunk:
   a. Perform an asynchronous Device-to-Host copy using cudaMemcpyAsync on a stream.
   b. Prepare a struct (CallbackArgs) containing the chunk index, size, and pointer to the destination buffer.
   c. Use cudaLaunchHostFunc (or the older cudaStreamAddCallback) to register a host callback that will be invoked once the copy is complete.
4. The host callback processes the chunk (e.g., computes a sum) and prints the result.
5. Synchronize all streams after launching all operations.
6. Clean up resources.
This approach allows the host to start processing each chunk as soon as its data is ready, while the GPU continues copying the next chunks, thus overlapping communication and computation for better pipeline efficiency.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cstdlib>

// Structure to hold callback arguments
struct CallbackArgs {
    int chunk_idx;
    int chunk_size;
    float* dst_ptr;
};

// Host callback invoked after the DtoH copy completes
void __stdcall hostCallback(cudaStream_t stream, cudaError_t status, void* userData) {
    CallbackArgs* args = static_cast<CallbackArgs*>(userData);
    if (status != cudaSuccess) {
        std::cerr << "Callback error for chunk " << args->chunk_idx
                  << ": " << cudaGetErrorString(status) << std::endl;
        return;
    }
    // Simple host-side processing: compute sum of the chunk
    float sum = 0.0f;
    for (int i = 0; i < args->chunk_size; ++i) {
        sum += args->dst_ptr[i];
    }
    std::cout << "Chunk " << args->chunk_idx
              << " processed, sum = " << sum << std::endl;
}

// Simple kernel to initialize device array
__global__ void initKernel(float* d_data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_data[idx] = static_cast<float>(idx);
    }
}

int main() {
    const int N = 1 << 20;          // Total number of elements (1,048,576)
    const int chunkSize = 1 << 18;  // Chunk size (262,144)
    const int numChunks = (N + chunkSize - 1) / chunkSize;
    const int numStreams = 4;       // Number of concurrent streams

    // Allocate host buffer
    float* h_dst = static_cast<float*>(malloc(N * sizeof(float)));
    if (!h_dst) {
        std::cerr << "Failed to allocate host memory." << std::endl;
        return EXIT_FAILURE;
    }

    // Allocate device buffer
    float* d_data = nullptr;
    cudaMalloc(&d_data, N * sizeof(float));

    // Initialize device array
    initKernel<<<(N + 255) / 256, 256>>>(d_data, N);
    cudaDeviceSynchronize();

    // Create streams
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Allocate callback argument array
    CallbackArgs* argsArray = new CallbackArgs[numChunks];

    // Launch multi-chunk copy and callbacks
    for (int chunk = 0; chunk < numChunks; ++chunk) {
        int offset = chunk * chunkSize;
        int currChunkSize = std::min(chunkSize, N - offset);
        float* dstPtr = h_dst + offset;

        // Asynchronous DtoH copy
        cudaMemcpyAsync(dstPtr, d_data + offset,
                        currChunkSize * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        streams[chunk % numStreams]);

        // Prepare callback arguments
        argsArray[chunk].chunk_idx = chunk;
        argsArray[chunk].chunk_size = currChunkSize;
        argsArray[chunk].dst_ptr = dstPtr;

        // Register host callback
        cudaLaunchHostFunc(streams[chunk % numStreams], hostCallback,
                           &argsArray[chunk]);
    }

    // Wait for all streams to finish
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    // Clean up
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_data);
    free(h_dst);
    delete[] argsArray;

    return 0;
}
```