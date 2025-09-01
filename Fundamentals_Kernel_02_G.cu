```c
/*
Write a kernel where only thread 0 of a block prints a message (`if (threadIdx.x == 0) { ... }`). Launch with 128 threads.

My thinking:
- The requirement is to write a CUDA kernel that prints only from the first thread of each block.
- The condition `if (threadIdx.x == 0)` ensures only the first thread in each block executes the `printf`.
- For simplicity, we can launch a single block with 128 threads. This satisfies the “Launch with 128 threads” requirement
  while still demonstrating that only thread 0 prints.
- The kernel uses device `printf`, which requires a compute capability of at least 2.0. The code includes basic
  error checking after the kernel launch and after `cudaDeviceSynchronize()` to catch any launch or runtime errors.
- The host code will call the kernel with `<<<1, 128>>>`, meaning one block of 128 threads.
- The output will be a single line from thread 0.
- This example is self‑contained and can be compiled with `nvcc -o example example.cu` and run on a CUDA capable device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a kernel launch or API call
#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Kernel where only thread 0 of the block prints a message
__global__ void printFromThread0()
{
    if (threadIdx.x == 0) {
        printf("Hello from thread 0 of block %d!\\n", blockIdx.x);
    }
}

int main(void)
{
    // Launch one block with 128 threads
    printFromThread0<<<1, 128>>>();
    CHECK_CUDA(cudaGetLastError());      // Check for launch errors
    CHECK_CUDA(cudaDeviceSynchronize()); // Wait for kernel to finish and check for runtime errors

    return 0;
}
```