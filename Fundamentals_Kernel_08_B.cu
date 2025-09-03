/*
Aim: Inside the kernel, calculate a flattened 1D block index from the 2D indices: int id_1d = blockIdx.y * gridDim.x + blockIdx.x;. Print this value.

Thinking:
1. The program demonstrates how to convert a 2‑D block coordinate (blockIdx.x, blockIdx.y) into a single linear index (id_1d) that uniquely identifies each block in a 2‑D grid.
2. The formula used is the standard row‑major ordering: `id_1d = blockIdx.y * gridDim.x + blockIdx.x;`.
3. To verify the calculation, the kernel will print the block coordinates and the resulting linear index using `printf`. CUDA kernels support `printf` from CUDA 2.0 onward.
4. Host code will set up a grid with a non‑trivial size (e.g., 4 columns × 3 rows) so that multiple blocks are launched. Each block has a single thread (blockDim set to (1,1,1)) because we only need one thread per block to perform the calculation.
5. After launching the kernel, the host will synchronize the device to ensure all output has been printed before exiting.
6. Error handling is added to catch any CUDA API errors.
7. The code is self‑contained and can be compiled with `nvcc` to produce an executable that prints the flattened block indices.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that calculates and prints the flattened block index
__global__ void printBlockId1D()
{
    // Compute the 1D block index from the 2D block indices
    int id_1d = blockIdx.y * gridDim.x + blockIdx.x;

    // Print the block coordinates and the computed 1D index
    printf("Block (%d, %d) -> id_1d: %d\n", blockIdx.x, blockIdx.y, id_1d);
}

// Helper macro for checking CUDA errors
#define CHECK_CUDA(call)                                        \
    do {                                                        \
        cudaError_t err = call;                                 \
        if (err != cudaSuccess) {                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while (0)

int main(void)
{
    // Define a 2D grid with 4 columns (x) and 3 rows (y)
    dim3 grid(4, 3);   // gridDim.x = 4, gridDim.y = 3
    dim3 block(1, 1);  // One thread per block

    // Launch the kernel
    printBlockId1D<<<grid, block>>>();

    // Wait for the kernel to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reset the device (optional cleanup)
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
