```cpp
/*
 * Aim of the program:
 * Launch the infinite loop kernel on a non-default stream. Does it still hang the application when you call `cudaDeviceSynchronize()`?
 *
 * Thinking:
 * The CUDA API function `cudaDeviceSynchronize()` blocks until all kernels on all streams have completed. Even if the infinite-loop kernel is launched on a user-created stream (non-default), `cudaDeviceSynchronize()` will still wait for it to finish. Since the kernel never terminates, the host will hang on this call. The code below demonstrates this behavior by launching a kernel that spins forever in a single thread on a non-default stream, then calling `cudaDeviceSynchronize()` to observe the hang. We also include basic error checking for clarity.
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

// Kernel that runs an infinite loop
__global__ void infiniteLoopKernel(void) {
    // Single thread executes an infinite loop
    while (true) {
        // Do nothing, just loop forever
    }
}

int main(void) {
    printf("Creating non-default CUDA stream...\n");
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    printf("Launching infinite loop kernel on non-default stream...\n");
    infiniteLoopKernel<<<1, 1, 0, stream>>>();
    CHECK_CUDA(cudaGetLastError());

    printf("Calling cudaDeviceSynchronize() - this should hang...\n");
    CHECK_CUDA(cudaDeviceSynchronize()); // This will block indefinitely

    // This line will never be reached
    printf("cudaDeviceSynchronize() returned, which should not happen.\n");

    // Cleanup (unreachable in this demo)
    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}
```