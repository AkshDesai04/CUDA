```cuda
/*
Aim: Write a program that processes a large dataset by breaking it into N chunks.
Perform the following sequence N times in a loop on the default stream:
(1) Copy HtoD, (2) Execute Kernel, (3) Copy DtoH. Time the total operation.

Thinking:
- The program will be written in CUDA C (.cu file) and compiled with nvcc.
- We'll define a total dataset size and a chunk size, compute N = total / chunk.
- The host will hold a large array of floats; the device will have a buffer the size of a chunk.
- A simple kernel will add 1.0f to each element (to keep the example trivial but measurable).
- The main loop will iterate N times, performing:
    * cudaMemcpy to copy one chunk from host to device,
    * kernel launch on the default stream,
    * cudaMemcpy to copy the processed chunk back to host.
- Timing will be performed using cudaEvent_t to capture GPU wall‑clock time from before the loop to after the loop.
- Basic error checking is added via a macro to keep code clean.
- The program prints the measured time in milliseconds.
- At the end, all allocated memory is freed.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

// Simple kernel that adds 1.0f to each element
__global__ void addOne(float *d_data, size_t numElements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        d_data[idx] += 1.0f;
    }
}

int main(void) {
    // Parameters
    const size_t totalSize = 1 << 28;      // 256M floats (~1GB)
    const size_t chunkSize  = 1 << 24;      // 16M floats (~64MB)
    const size_t numChunks  = totalSize / chunkSize;
    const size_t bytesChunk = chunkSize * sizeof(float);

    // Allocate host memory
    float *h_data = (float*)malloc(totalSize * sizeof(float));
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (size_t i = 0; i < totalSize; ++i) {
        h_data[i] = static_cast<float>(i % 1000);
    }

    // Allocate device memory for one chunk
    float *d_chunk;
    CHECK_CUDA(cudaMalloc((void**)&d_chunk, bytesChunk));

    // Setup CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Main processing loop over chunks
    for (size_t chunk = 0; chunk < numChunks; ++chunk) {
        size_t offset = chunk * chunkSize;

        // Copy chunk from host to device (HtoD)
        CHECK_CUDA(cudaMemcpy(d_chunk, h_data + offset, bytesChunk,
                              cudaMemcpyHostToDevice));

        // Determine grid size
        int threadsPerBlock = 256;
        int blocksPerGrid = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;

        // Launch kernel
        addOne<<<blocksPerGrid, threadsPerBlock>>>(d_chunk, chunkSize);

        // Check for kernel launch errors
        CHECK_CUDA(cudaGetLastError());

        // Copy result back to host (DtoH)
        CHECK_CUDA(cudaMemcpy(h_data + offset, d_chunk, bytesChunk,
                              cudaMemcpyDeviceToHost));
    }

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Compute elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Total time for processing %zu chunks: %f ms\n",
           numChunks, milliseconds);

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_chunk));
    free(h_data);

    return EXIT_SUCCESS;
}
```