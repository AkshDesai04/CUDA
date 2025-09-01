```c
/*
Aim of the program:
Write a kernel where only thread 0 of a block prints a message (if (threadIdx.x == 0) { ... }). Launch with 128 threads.

Thinking:
- The requirement is straightforward: a CUDA kernel where only thread 0 of a block executes a printf.
- CUDA device code supports printf starting with compute capability 2.0, which is available on most GPUs.
- We'll use a single block with 128 threads, as specified. Thread 0 will check its threadIdx.x and print a message including the block index.
- After kernel launch, the host should call cudaDeviceSynchronize() to ensure the device has finished executing before the program exits.
- Include necessary headers, error checking macro for brevity, and compile with nvcc. The code is self-contained and ready to run.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                  \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));    \
            return EXIT_FAILURE;                                                \
        }                                                                       \
    } while (0)

__global__ void printFromThread0()
{
    if (threadIdx.x == 0) {
        printf("Hello from block %d, thread 0!\n", blockIdx.x);
    }
}

int main(void)
{
    // Launch one block with 128 threads
    printFromThread0<<<1, 128>>>();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Optional: indicate completion
    printf("Kernel launch complete.\n");
    return EXIT_SUCCESS;
}
```