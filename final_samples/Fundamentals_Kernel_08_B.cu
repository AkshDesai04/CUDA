```c
/*
Inside the kernel, calculate a flattened 1D block index from the 2D indices: `int id_1d = blockIdx.y * gridDim.x + blockIdx.x;`. Print this value.

Thinking:
- The task is to write a CUDA program that demonstrates how to compute a flattened 1D block index from the 2D block indices (blockIdx.x, blockIdx.y) and print that value.
- The kernel should simply calculate `int id_1d = blockIdx.y * gridDim.x + blockIdx.x;` and use `printf` to output the result along with the block coordinates.
- We will launch the kernel with a grid that has multiple blocks in both x and y dimensions to showcase different id_1d values.
- Host code will set up the grid dimensions, launch the kernel, synchronize, and check for errors.
- The code will include necessary headers and use CUDA runtime API functions.
- It will be written in C style, compile with nvcc as a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that calculates and prints the flattened 1D block index
__global__ void compute_id1d()
{
    int id_1d = blockIdx.y * gridDim.x + blockIdx.x;
    printf("Block (%d, %d) => Flattened 1D id_1d = %d\n", blockIdx.x, blockIdx.y, id_1d);
}

// Utility macro for error checking
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

int main(void)
{
    // Define grid and block dimensions
    dim3 gridDim(4, 3);  // 4 blocks in x, 3 blocks in y
    dim3 blockDim(1, 1); // 1 thread per block (not relevant for this example)

    // Launch the kernel
    compute_id1d<<<gridDim, blockDim>>>();

    // Wait for GPU to finish before accessing on host
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reset the device (optional)
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
```