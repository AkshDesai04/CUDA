```c
/* 
Launch a kernel with a 2D block of threads, for example, 8x8 threads. Each thread should print its 2D thread index (threadIdx.x, threadIdx.y).

Thoughts:
- We need a CUDA kernel that runs on the device and prints the 2D indices.
- CUDA device code can use printf; it will print to stdout once the kernel finishes.
- The block dimensions will be set to dim3(8, 8) to create an 8x8 grid of threads.
- Launch configuration: one block (1,1,1) for simplicity.
- After launching, we need to synchronize the device to ensure all prints complete before the host exits.
- Include simple error checking macro to catch kernel launch failures.
- The program must be self‑contained and compile with nvcc as a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple error checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",    \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),         \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Device kernel that prints its 2D thread indices
__global__ void printThreadIdx()
{
    // Print the thread indices of the current thread
    printf("Thread (%d, %d)\n", threadIdx.x, threadIdx.y);
}

int main(void)
{
    // Define block dimensions (8x8)
    dim3 blockDim(8, 8, 1);

    // Launch kernel with a single block
    printThreadIdx<<<1, blockDim>>>();

    // Check for any launch errors
    CUDA_CHECK(cudaGetLastError());

    // Wait for the device to finish all work
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reset the device (optional cleanup)
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
```