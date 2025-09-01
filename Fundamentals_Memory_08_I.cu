```cpp
/*
Use `cudaMemcpyAsync` on a stream and record events around it to time the asynchronous transfer.

The aim of the program is to demonstrate how to use cudaMemcpyAsync in conjunction with CUDA streams
and CUDA events to perform an asynchronous memory transfer from host to device, while timing that
transfer.  The program follows these steps:

1. Allocate a pinned host buffer using cudaMallocHost to enable efficient asynchronous copies.
2. Allocate a device buffer with cudaMalloc.
3. Initialize the host buffer with deterministic data so that the transfer can be verified if desired.
4. Create a CUDA stream.  All operations submitted to this stream execute in order, but can overlap
   with operations in other streams or with the host.
5. Create two CUDA events, `start` and `stop`.  Events capture timestamps relative to the stream
   in which they are recorded.
6. Record the `start` event on the stream, then issue a cudaMemcpyAsync to transfer the host buffer
   to the device buffer, specifying the same stream so the copy is queued after the `start`
   event.
7. Record the `stop` event on the stream immediately after the memcpy.  Because the event is
   queued after the copy, its timestamp will be after the copy has finished.
8. Synchronize on the `stop` event (or the stream) to ensure the transfer has completed.
9. Use cudaEventElapsedTime to compute the elapsed time in milliseconds between the two events.
10. Print the measured transfer time and perform any optional verification.
11. Clean up: free host/device memory, destroy events, destroy the stream.

The resulting output reports the time taken for the asynchronous transfer, demonstrating the
performance difference between synchronous and asynchronous copies when using pinned memory.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Simple macro to check CUDA errors
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",    \
                    __FILE__, __LINE__, err, cudaGetErrorString(err), #call);\
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void) {
    const size_t N = 1 << 24;          // Number of elements (16M doubles ~ 128MB)
    const size_t bytes = N * sizeof(double);

    // Allocate pinned host memory
    double *h_array = NULL;
    CHECK_CUDA(cudaMallocHost((void **)&h_array, bytes));

    // Initialize host array
    for (size_t i = 0; i < N; ++i) {
        h_array[i] = static_cast<double>(i);
    }

    // Allocate device memory
    double *d_array = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_array, bytes));

    // Create a stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Create events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record start event
    CHECK_CUDA(cudaEventRecord(start, stream));

    // Asynchronous memcpy from host to device on the created stream
    CHECK_CUDA(cudaMemcpyAsync(d_array, h_array, bytes, cudaMemcpyHostToDevice, stream));

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop, stream));

    // Wait for the event to complete (ensures copy has finished)
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Compute elapsed time in milliseconds
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    printf("Asynchronous cudaMemcpyAsync time: %f ms\n", ms);

    // Optional: verify transfer by copying back and checking a few values
    /*
    double *h_check = NULL;
    CHECK_CUDA(cudaMallocHost((void **)&h_check, bytes));
    CHECK_CUDA(cudaMemcpy(h_check, d_array, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < 10; ++i) {
        printf("h_check[%zu] = %f\n", i, h_check[i]);
    }
    CHECK_CUDA(cudaFreeHost(h_check));
    */

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_array));
    CHECK_CUDA(cudaFreeHost(h_array));

    return 0;
}
```