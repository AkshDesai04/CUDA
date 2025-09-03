```cpp
/*
Aim of the program: Modify the kernel so that threads with `threadIdx.x == threadIdx.y` (on the diagonal) print a special message.

Thinking:
- The task is to write a CUDA program that demonstrates how to detect diagonal threads inside a 2D thread block.
- The kernel will be launched with a 2D grid and a 2D block. Each thread will check if its x and y indices are equal.
- If they are equal, the thread will print a special message indicating it is on the diagonal. Otherwise, it prints a regular message.
- Device-side printf is available for compute capability >= 2.0. We will use it for demonstration.
- The host code will set up a simple 8x8 block and launch the kernel once.
- Synchronization will be handled via cudaDeviceSynchronize, and any errors will be reported.
- The entire program will be a single .cu file, ready to compile with `nvcc -arch=sm_61 test.cu -o test` (or any supported arch).
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                       \
    } while (0)

__global__ void diagKernel()
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // For demonstration, each thread prints its position
    if (x == y) {
        // Diagonal thread: print special message
        printf("[Block (%d,%d)] Thread (%d,%d) is on the DIAGONAL!\n",
               bx, by, x, y);
    } else {
        printf("[Block (%d,%d)] Thread (%d,%d) is off-diagonal.\n",
               bx, by, x, y);
    }
}

int main(void)
{
    // Define block and grid dimensions
    dim3 blockDim(8, 8, 1); // 8x8 threads per block
    dim3 gridDim(2, 2, 1);  // 2x2 blocks

    // Launch kernel
    diagKernel<<<gridDim, blockDim>>>();

    // Wait for GPU to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reset device (optional)
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
```