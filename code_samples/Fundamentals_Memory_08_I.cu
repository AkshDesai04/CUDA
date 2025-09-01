/*
Aim of the program: Use `cudaMemcpyAsync` on a stream and record events around it to time the asynchronous transfer.

Thought process:
1. The task is to create a simple CUDA C program that demonstrates asynchronous memory copying using cudaMemcpyAsync.
2. We need to set up a CUDA stream, allocate host and device memory, and perform a copy from host to device using the stream.
3. To time the asynchronous transfer, we will use CUDA events (cudaEvent_t). We will record a start event immediately before launching the async copy, and a stop event immediately after the launch.
4. After launching the copy, we must synchronize the stream (or the events) to ensure the copy has completed before we compute elapsed time.
5. We'll then compute the elapsed time in milliseconds using cudaEventElapsedTime and print it.
6. Throughout the code, we should handle errors with a simple macro that checks return codes.
7. Finally, we should free all allocated resources (device memory, host memory, events, stream) and exit cleanly.
8. The program will be self-contained and compile as a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorName(err), cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                          \
    } while (0)

int main(void) {
    const size_t N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_src = (float*)malloc(size);
    if (!h_src) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    // Initialize host data
    for (size_t i = 0; i < N; ++i) h_src[i] = (float)i;

    // Allocate device memory
    float *d_dst = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_dst, size));

    // Create a stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start, stream));

    // Launch async copy from host to device
    CUDA_CHECK(cudaMemcpyAsync(d_dst, h_src, size, cudaMemcpyHostToDevice, stream));

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop, stream));

    // Wait for the events to complete
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Compute elapsed time
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("Asynchronous copy time: %f ms\n", ms);

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_dst));
    free(h_src);

    return EXIT_SUCCESS;
}
