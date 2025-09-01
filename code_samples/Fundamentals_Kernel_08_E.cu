/*
Aim: Combine with 2D blocks. Launch a 2x2 grid of 4x4 blocks. Have each thread calculate and print its global 2D index: int global_x = blockIdx.x * blockDim.x + threadIdx.x; and int global_y = ... .

Thinking:
- The goal is to demonstrate a simple CUDA kernel that uses a 2‑dimensional grid and block layout.
- The grid dimensions are 2x2 and each block is 4x4, so there are a total of 16 blocks and 16 threads per block, i.e. 256 threads.
- Each thread needs to compute its global coordinates from its block and thread indices and print them using `printf`.
- CUDA `printf` is available on devices of compute capability 2.0 and higher, so we will assume a modern GPU.
- We will write a small `main` that launches the kernel, synchronizes, and checks for errors.
- The kernel will be defined as `__global__ void printGlobalIndices()`. Inside, we calculate `global_x` and `global_y` and call `printf`.
- The host code will configure the launch with `dim3 grid(2, 2); dim3 block(4, 4);` and launch the kernel.
- Error checking macros will be used to catch any CUDA runtime errors.
- Finally, we will print a completion message and exit.

This program can be compiled with `nvcc -arch=sm_20 indices.cu -o indices` and run on a suitable GPU.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__global__ void printGlobalIndices(void) {
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    printf("Thread (%d, %d) has global index (%d, %d)\n",
           threadIdx.x, threadIdx.y, global_x, global_y);
}

int main(void) {
    dim3 grid(2, 2);
    dim3 block(4, 4);

    // Launch kernel
    printGlobalIndices<<<grid, block>>>();
    CUDA_CHECK(cudaGetLastError());

    // Wait for GPU to finish before accessing on host
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("All threads have reported their global indices.\n");
    return 0;
}
