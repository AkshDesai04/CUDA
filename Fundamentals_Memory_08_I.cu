```cuda
/*
Use `cudaMemcpyAsync` on a stream and record events around it to time the asynchronous transfer.
In this program we allocate pinned host memory and device memory for a large array of integers.
We then create a CUDA stream and two CUDA events (startEvent and stopEvent). 
Before issuing the asynchronous memory copy we record startEvent, and immediately after we record stopEvent. 
We then synchronize on the stream (or the stopEvent) to ensure the copy has finished.
Finally, we query the elapsed time between the two events using cudaEventElapsedTime, which gives the time in milliseconds that the asynchronous transfer took. 
The program also demonstrates a simple error checking macro and reports the measured transfer time and the transfer throughput in GB/s. 
The logic is straightforward: allocate memory, initialize host data, create stream and events, time the async memcpy, synchronize, compute elapsed time, clean up.
*/

#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

int main() {
    const size_t N = 10 * 1024 * 1024; // 10 million integers
    const size_t size = N * sizeof(int);

    // Allocate pinned host memory for source and destination
    int *h_src, *h_dst;
    CHECK_CUDA(cudaHostAlloc((void**)&h_src, size, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc((void**)&h_dst, size, cudaHostAllocDefault));

    // Initialize source data
    for (size_t i = 0; i < N; ++i) {
        h_src[i] = static_cast<int>(i);
    }

    // Allocate device memory
    int *d_buf;
    CHECK_CUDA(cudaMalloc((void**)&d_buf, size));

    // Create a stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Create events for timing
    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));

    // Record start event
    CHECK_CUDA(cudaEventRecord(startEvent, stream));

    // Asynchronously copy from host to device
    CHECK_CUDA(cudaMemcpyAsync(d_buf, h_src, size, cudaMemcpyHostToDevice, stream));

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stopEvent, stream));

    // Wait for the stream to complete (ensures copy finished)
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Compute elapsed time
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, startEvent, stopEvent));

    // Calculate throughput in GB/s
    float gbTransferred = (float)size / (1024.0f * 1024.0f * 1024.0f);
    float throughput = gbTransferred / (ms / 1000.0f);

    printf("Asynchronous memcpy transferred %.2f GB in %.3f ms (throughput: %.2f GB/s)\n",
           gbTransferred, ms, throughput);

    // Clean up
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaFreeHost(h_src));
    CHECK_CUDA(cudaFreeHost(h_dst));

    return 0;
}
```