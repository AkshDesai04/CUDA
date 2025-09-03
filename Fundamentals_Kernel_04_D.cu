/*
Aim: Modify the kernel so that only blocks with an odd index print their ID.

This program demonstrates how to conditionally print from a CUDA kernel based on the
block index. The kernel is launched with a specified number of blocks and a
fixed number of threads per block. Inside the kernel, we check the block index
using `blockIdx.x`. If the index is odd (`blockIdx.x % 2 == 1` or equivalently
`blockIdx.x & 1`), the thread prints its block ID using the device `printf`
function. All other blocks perform no printing. The host code sets up the
kernel launch parameters, synchronizes the device to ensure all printf
output has been flushed, and performs basic error checking.

Key points:
- `__global__` kernel launches with `printf` support.
- Conditional `if ((blockIdx.x & 1) == 1)` ensures only odd blocks execute
  the printing statement.
- Host uses `cudaDeviceSynchronize()` to wait for all kernel execution and
  device printing to complete before exiting.
- Error checking after kernel launch and synchronization helps catch any
  launch or runtime errors.

The program can be compiled with nvcc:
    nvcc -o odd_block_print odd_block_print.cu
and run:
    ./odd_block_print
The expected output will list block IDs 1, 3, 5, 7 (for 8 blocks total).
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints the block ID only if the block index is odd
__global__ void printOddBlockIds()
{
    // Each block checks its own index; no need to restrict to a particular thread
    if ((blockIdx.x & 1) == 1) {
        printf("Block ID: %d\n", blockIdx.x);
    }
}

// Helper macro for error checking
#define CUDA_CHECK(call)                                 \
    do {                                                 \
        cudaError_t err = call;                          \
        if (err != cudaSuccess) {                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                         \
        }                                                \
    } while (0)

int main(void)
{
    // Define grid and block dimensions
    const int threadsPerBlock = 256;
    const int numBlocks = 8;  // Example: 8 blocks (0-7)

    // Launch the kernel
    printOddBlockIds<<<numBlocks, threadsPerBlock>>>();

    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());

    // Wait for all kernels to finish and flush printf output
    CUDA_CHECK(cudaDeviceSynchronize());

    // Optional: reset device (clean up)
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
