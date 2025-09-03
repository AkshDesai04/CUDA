/*
Calculate the speedup factor (Sequential Time / Streamed Time)

I want to write a CUDA program that measures the execution time of a simple operation (vector addition) performed sequentially and with multiple streams, then compute the speedup factor as Sequential Time divided by Streamed Time.

Design decisions:
1. Use a simple kernel that adds a constant value to each element of an array.
2. For the sequential case, launch a single kernel that processes the entire array on the default stream.
3. For the streamed case, split the array into equal-sized chunks and launch a kernel for each chunk on its own CUDA stream. This allows potential overlap of kernel execution if the device supports concurrent kernels.
4. Measure execution time using cudaEvent_t start and stop events. The sequential case will have two events (before and after the kernel). The streamed case will record the start event before launching all kernels and a stop event after synchronizing all streams.
5. Compute times in milliseconds, then compute the speedup factor as seqTime / streamTime.
6. Print out both times and the speedup.

Potential pitfalls:
- The actual concurrency depends on the GPU's capability; on many devices kernels will still execute serially but the code will illustrate the intended method.
- Ensure proper error checking for CUDA API calls.
- Use a reasonable number of streams (e.g., 4) and chunk size to fit within device memory.

The program will be self-contained and compile with nvcc. */

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that adds 1.0f to each element in the specified range.
__global__ void addKernel(float *d_arr, int offset, int chunkSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int globalIdx = offset + idx;
    if (idx < chunkSize)
        d_arr[globalIdx] += 1.0f;
}

// Macro for checking CUDA errors.
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

int main()
{
    const int N = 1 << 20;          // 1M elements
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Host allocation and initialization
    float *h_arr = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i)
        h_arr[i] = 0.0f; // initial value

    // Device allocation
    float *d_arr;
    CUDA_CHECK(cudaMalloc((void **)&d_arr, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice));

    // Sequential execution
    cudaEvent_t seq_start, seq_stop;
    CUDA_CHECK(cudaEventCreate(&seq_start));
    CUDA_CHECK(cudaEventCreate(&seq_stop));

    CUDA_CHECK(cudaEventRecord(seq_start, 0));
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, 0, N);
    CUDA_CHECK(cudaEventRecord(seq_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(seq_stop));

    float seq_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&seq_time_ms, seq_start, seq_stop));

    // Streamed execution
    const int numStreams = 4;
    int chunkSize = N / numStreams;
    int remainder = N % numStreams;

    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; ++i)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

    cudaEvent_t stream_start, stream_stop;
    CUDA_CHECK(cudaEventCreate(&stream_start));
    CUDA_CHECK(cudaEventCreate(&stream_stop));

    // Reset array to zeros for fair comparison
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(stream_start, 0));
    int offset = 0;
    for (int i = 0; i < numStreams; ++i)
    {
        int currentChunkSize = chunkSize + (i < remainder ? 1 : 0);
        addKernel<<<(currentChunkSize + threadsPerBlock - 1) / threadsPerBlock,
                    threadsPerBlock, 0, streams[i]>>>(d_arr, offset, currentChunkSize);
        offset += currentChunkSize;
    }

    // Wait for all streams to finish
    for (int i = 0; i < numStreams; ++i)
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));

    CUDA_CHECK(cudaEventRecord(stream_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stream_stop));

    float stream_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&stream_time_ms, stream_start, stream_stop));

    // Compute speedup
    float speedup = seq_time_ms / stream_time_ms;

    // Output results
    printf("Sequential Time (ms): %f\n", seq_time_ms);
    printf("Streamed Time (ms):   %f\n", stream_time_ms);
    printf("Speedup (Seq/Stream): %f\n", speedup);

    // Cleanup
    for (int i = 0; i < numStreams; ++i)
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    CUDA_CHECK(cudaEventDestroy(seq_start));
    CUDA_CHECK(cudaEventDestroy(seq_stop));
    CUDA_CHECK(cudaEventDestroy(stream_start));
    CUDA_CHECK(cudaEventDestroy(stream_stop));
    CUDA_CHECK(cudaFree(d_arr));
    free(h_arr);

    return 0;
}
