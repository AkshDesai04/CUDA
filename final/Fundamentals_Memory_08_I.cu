```cuda
/*
 * Use `cudaMemcpyAsync` on a stream and record events around it to time the asynchronous transfer.
 *
 * My thinking while writing this code:
 * 1. The goal is to demonstrate how to use cudaMemcpyAsync along with a CUDA stream to perform
 *    asynchronous data transfer from host to device, and to measure the time taken for that
 *    transfer using CUDA events.
 *
 * 2. I will allocate a small array on the host, fill it with some data, then allocate a
 *    corresponding buffer on the device. A CUDA stream will be created to enqueue the
 *    asynchronous copy operation.
 *
 * 3. Two CUDA events (start and stop) will be recorded immediately before and after the
 *    cudaMemcpyAsync call. After the copy operation, I will wait for the stop event to
 *    complete and then query the elapsed time. This gives the wall‑clock time for the
 *    asynchronous transfer. The code will also use cudaDeviceSynchronize to ensure
 *    the transfer has finished before printing the result.
 *
 * 4. To make the code robust, I will wrap CUDA API calls in an error‑checking macro that
 *    prints an informative message and exits if a call fails.
 *
 * 5. The program will print the size of the transfer in bytes and the measured transfer
 *    time in milliseconds. This illustrates that cudaMemcpyAsync can overlap with other
 *    work on the stream, but here we only measure the transfer time.
 *
 * 6. The code is written in pure C++ (CUDA C) and can be compiled with nvcc as a
 *    single .cu file. No external dependencies are required.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void) {
    const size_t N = 1 << 20;          // Number of elements (1M)
    const size_t bytes = N * sizeof(float);

    // Allocate and initialize host memory
    float *h_data = (float *)malloc(bytes);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    for (size_t i = 0; i < N; ++i) h_data[i] = static_cast<float>(i);

    // Allocate device memory
    float *d_data;
    CHECK_CUDA(cudaMalloc((void **)&d_data, bytes));

    // Create a CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Create events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record the start event
    CHECK_CUDA(cudaEventRecord(start, stream));

    // Perform asynchronous copy HtoD
    CHECK_CUDA(cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, stream));

    // Record the stop event
    CHECK_CUDA(cudaEventRecord(stop, stream));

    // Wait for the copy to finish
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // Print results
    printf("Transferred %zu bytes from host to device in %f ms (stream=%p)\n",
           bytes, ms, (void *)stream);

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);

    return EXIT_SUCCESS;
}
```