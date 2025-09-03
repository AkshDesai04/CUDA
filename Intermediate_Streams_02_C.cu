/*
The main loop will be more complex. A common pattern is to prime the pipeline, then loop: `for k=0 to N-1: stream = streams[k%2]; ... issue async work on stream... cudaStreamSynchronize(prev_stream); ... process result from prev_stream ...`

**Thinking**  
The aim is to demonstrate a double‑buffered CUDA pipeline using two streams to overlap data transfer, kernel execution, and CPU processing.  
We allocate pinned host memory for input and output, device buffers for two halves, and two CUDA streams.  
The program primes the pipeline by launching the first iteration on stream 0.  
For each subsequent iteration, it waits on the previous stream, processes the CPU‑side results, then issues async copy, kernel, and copy‑back for the next chunk on the current stream.  
Finally, after the loop, the last stream is synchronized and its results processed.  
A simple kernel that increments each element is used, and the CPU processing just sums the chunk and prints it.  
Error handling is provided via a helper macro.  
This demonstrates how to pipeline I/O, compute, and post‑processing efficiently with CUDA streams. 
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

__global__ void incKernel(int *d_data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) d_data[idx] += 1;
}

void processResult(int *h_buffer, int size, int chunkIdx)
{
    long long sum = 0;
    for (int i = 0; i < size; ++i)
        sum += h_buffer[i];
    printf("Chunk %d: sum = %lld\n", chunkIdx, sum);
}

int main(void)
{
    const int totalElements = 1 << 20; // 1M elements
    const int chunkSize     = 1 << 18; // 256K elements per chunk
    const int numChunks     = (totalElements + chunkSize - 1) / chunkSize;

    // Allocate pinned host memory
    int *h_input[2];
    int *h_output[2];
    for (int i = 0; i < 2; ++i) {
        CHECK_CUDA(cudaMallocHost((void **)&h_input[i], chunkSize * sizeof(int)));
        CHECK_CUDA(cudaMallocHost((void **)&h_output[i], chunkSize * sizeof(int)));
    }

    // Fill input with dummy data
    for (int i = 0; i < totalElements; ++i)
        h_input[0][i] = i; // Only first buffer used for input; we copy relevant portion

    // Allocate device memory for double buffering
    int *d_input[2];
    int *d_output[2];
    for (int i = 0; i < 2; ++i) {
        CHECK_CUDA(cudaMalloc((void **)&d_input[i], chunkSize * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void **)&d_output[i], chunkSize * sizeof(int)));
    }

    // Create two streams
    cudaStream_t streams[2];
    CHECK_CUDA(cudaStreamCreate(&streams[0]));
    CHECK_CUDA(cudaStreamCreate(&streams[1]));

    // Prime the pipeline: launch first chunk on stream 0
    int currSize = (totalElements < chunkSize) ? totalElements : chunkSize;
    CHECK_CUDA(cudaMemcpyAsync(d_input[0], h_input[0], currSize * sizeof(int),
                               cudaMemcpyHostToDevice, streams[0]]));
    int threadsPerBlock = 256;
    int blocks = (currSize + threadsPerBlock - 1) / threadsPerBlock;
    incKernel<<<blocks, threadsPerBlock, 0, streams[0]>>>(d_input[0], currSize);
    CHECK_CUDA(cudaMemcpyAsync(h_output[0], d_output[0], currSize * sizeof(int),
                               cudaMemcpyDeviceToHost, streams[0]]));
    CHECK_CUDA(cudaGetLastError());

    // Main loop
    for (int k = 1; k < numChunks; ++k) {
        int prevStream = streams[(k - 1) % 2];
        int currStream = streams[k % 2];

        // Wait for previous stream to finish before processing its result
        CHECK_CUDA(cudaStreamSynchronize(prevStream));
        int prevSize = (k == 1) ? currSize : chunkSize;
        processResult(h_output[(k - 1) % 2], prevSize, k - 1);

        // Prepare next chunk
        int offset = k * chunkSize;
        int remaining = totalElements - offset;
        int thisSize = (remaining < chunkSize) ? remaining : chunkSize;
        // Copy input to device
        CHECK_CUDA(cudaMemcpyAsync(d_input[k % 2], h_input[0] + offset,
                                   thisSize * sizeof(int), cudaMemcpyHostToDevice,
                                   currStream));
        // Launch kernel
        blocks = (thisSize + threadsPerBlock - 1) / threadsPerBlock;
        incKernel<<<blocks, threadsPerBlock, 0, currStream>>>(d_input[k % 2], thisSize);
        // Copy result back
        CHECK_CUDA(cudaMemcpyAsync(h_output[k % 2], d_output[k % 2],
                                   thisSize * sizeof(int), cudaMemcpyDeviceToHost,
                                   currStream));
        CHECK_CUDA(cudaGetLastError());
    }

    // Process the last chunk
    int lastIdx = numChunks - 1;
    int lastStream = streams[lastIdx % 2];
    CHECK_CUDA(cudaStreamSynchronize(lastStream));
    int lastSize = totalElements - lastIdx * chunkSize;
    processResult(h_output[lastIdx % 2], lastSize, lastIdx);

    // Clean up
    for (int i = 0; i < 2; ++i) {
        CHECK_CUDA(cudaFreeHost(h_input[i]));
        CHECK_CUDA(cudaFreeHost(h_output[i]));
        CHECK_CUDA(cudaFree(d_input[i]));
        CHECK_CUDA(cudaFree(d_output[i]));
    }
    CHECK_CUDA(cudaStreamDestroy(streams[0]));
    CHECK_CUDA(cudaStreamDestroy(streams[1]]);

    return 0;
}
