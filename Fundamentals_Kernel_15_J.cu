```cpp
/* 
Aim: Launch the infinite loop kernel on a non-default stream. Does it still hang the application when you call cudaDeviceSynchronize()?

Thinking:
1. We need a CUDA kernel that never exits. A simple `while(true) {}` loop inside the kernel will keep a thread busy forever. 
2. The kernel will be launched on a stream other than the default stream (stream 0). We create a new stream with `cudaStreamCreate`.
3. After launching the kernel, we call `cudaDeviceSynchronize()`. According to CUDA semantics, this blocks the host until all device work in all streams completes. Since the kernel never completes, `cudaDeviceSynchronize` will block forever, even though it is on a non-default stream.
4. To verify that the kernel was launched, we print a message from the host before the synchronization call. The program will hang at `cudaDeviceSynchronize()`, demonstrating that the non-default stream does not avoid the hang.
5. The code includes basic error checking via a macro to make debugging easier. The program can be compiled with `nvcc` and run on any CUDA-capable device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel that runs an infinite loop
__global__ void infiniteLoopKernel()
{
    // Each thread spins forever
    while (true) { }
}

int main()
{
    printf("Creating a non-default stream.\n");
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    dim3 grid(1);
    dim3 block(1);

    printf("Launching infinite-loop kernel on non-default stream.\n");
    infiniteLoopKernel<<<grid, block, 0, stream>>>();

    // Check for any launch errors
    CUDA_CHECK(cudaGetLastError());

    printf("Calling cudaDeviceSynchronize() - this will block forever.\n");
    CUDA_CHECK(cudaDeviceSynchronize()); // This will hang

    // The following code will never be reached
    printf("This line will never be executed.\n");
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}
```