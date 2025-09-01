```cpp
/*
Launch the infinite loop kernel on a non-default stream. Does it still hang the application when you call `cudaDeviceSynchronize()`?

Thinking:
- The purpose of this program is to demonstrate that launching a kernel that runs forever (infinite loop) on a stream other than the default stream (stream 0) will still cause a call to `cudaDeviceSynchronize()` to block the host indefinitely.
- We create a non-default stream with `cudaStreamCreate()`, launch a kernel that never exits (`while(true){}`) into that stream, and then call `cudaDeviceSynchronize()`.
- Since the kernel is stuck in an infinite loop, the device will never finish the kernel, so `cudaDeviceSynchronize()` will never return. The program will hang at that point, confirming that the stream property does not affect the blocking behavior of `cudaDeviceSynchronize()`.
- We include basic error checking for stream creation, kernel launch, and the synchronize call, but the program will not reach the final print because the sync never returns.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Kernel that runs an infinite loop
__global__ void infinite_loop()
{
    while (true)
    {
        // Do nothing, just spin
    }
}

// Helper macro for checking CUDA errors
#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main()
{
    printf("Creating a non-default stream...\n");
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    printf("Launching infinite loop kernel on the non-default stream...\n");
    infinite_loop<<<1, 1, 0, stream>>>();
    CHECK_CUDA(cudaGetLastError());

    printf("Kernel launched. Calling cudaDeviceSynchronize() to wait for completion...\n");
    CHECK_CUDA(cudaDeviceSynchronize()); // This call will block forever

    // This line will never be reached
    printf("This message will not be printed because the application hangs at synchronize.\n");

    // Clean up (unreachable in practice)
    cudaStreamDestroy(stream);
    return 0;
}
```