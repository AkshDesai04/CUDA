```cpp
/*
Aim of the program:
Launch a 4x4 grid. Pass a 2D array (flattened to 1D) int* data. Have each thread write its flattened 1D block index to the corresponding location in the array.

Thinking:
- The grid is 4 blocks in the x-dimension and 4 blocks in the y-dimension, resulting in 16 blocks.
- Each block will contain a single thread (default block size 1x1). 
- The kernel will compute a flattened block index: idx = blockIdx.y * gridDim.x + blockIdx.x.
- Each thread writes this index to the array element at the same flattened index.
- The array therefore has 16 elements.
- Allocate device memory, launch kernel, copy back, and print the results.
- Basic CUDA error checking is included.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple macro for CUDA error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel: each thread writes its flattened block index to data[idx]
__global__ void writeBlockIdx(int *data) {
    // Flattened 1D index of the block
    int idx = blockIdx.y * gridDim.x + blockIdx.x;
    data[idx] = idx;
}

int main(void) {
    const int GRID_X = 4;
    const int GRID_Y = 4;
    const int ARRAY_SIZE = GRID_X * GRID_Y; // 16 elements

    int *d_data = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_data, ARRAY_SIZE * sizeof(int)));

    // Launch kernel with 4x4 grid of blocks, each block with 1 thread
    writeBlockIdx<<<dim3(GRID_X, GRID_Y), 1>>>(d_data);
    CUDA_CHECK(cudaGetLastError()); // Check kernel launch error
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel completion

    // Copy data back to host
    int h_data[ARRAY_SIZE];
    CUDA_CHECK(cudaMemcpy(h_data, d_data, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Block indices written to array:\n");
    for (int y = 0; y < GRID_Y; ++y) {
        for (int x = 0; x < GRID_X; ++x) {
            int idx = y * GRID_X + x;
            printf("%2d ", h_data[idx]);
        }
        printf("\n");
    }

    CUDA_CHECK(cudaFree(d_data));
    return 0;
}
```