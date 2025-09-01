```cuda
/*
Aim: Launch an 8x8 block. Pass a 2D array (flattened to 1D) `int* data`. Have each thread write its flattened 1D index to the corresponding location in the array.

Thinking:
- The goal is to demonstrate basic CUDA thread indexing in a single 8x8 block.
- We need a kernel that receives a pointer to a flattened 1D array of ints.
- Each thread must compute its own flattened 1D index. Since the block is 8x8, the index is
    idx = threadIdx.y * blockDim.x + threadIdx.x
  where threadIdx.x ranges from 0..7 and threadIdx.y ranges from 0..7.
- The kernel writes the computed index back into the array at position `idx`.
- On the host side:
  - Allocate a host array of size 64 (8*8).
  - Allocate a device array of the same size.
  - Launch the kernel with a single block of size 8x8: <<<1, dim3(8,8)>>>
  - Copy the result back to the host.
  - Print the array in an 8x8 grid format to verify correctness.
- Include error checking after kernel launch to catch any launch failures.
- Use standard C headers and the CUDA runtime header.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that writes each thread's flattened index into the array
__global__ void writeIndex(int* data)
{
    // Compute 1D flattened index from 2D thread coordinates
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    // Write the index into the corresponding array element
    data[idx] = idx;
}

int main(void)
{
    const int WIDTH = 8;
    const int HEIGHT = 8;
    const int SIZE = WIDTH * HEIGHT;
    const size_t bytes = SIZE * sizeof(int);

    // Allocate host memory
    int* h_data = (int*)malloc(bytes);
    if (h_data == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    int* d_data = NULL;
    cudaError_t err = cudaMalloc((void**)&d_data, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        free(h_data);
        return EXIT_FAILURE;
    }

    // Launch kernel with a single 8x8 block
    dim3 blockDim(WIDTH, HEIGHT, 1);
    dim3 gridDim(1, 1, 1);
    writeIndex<<<gridDim, blockDim>>>(d_data);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return EXIT_FAILURE;
    }

    // Copy result back to host
    err = cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from device to host: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return EXIT_FAILURE;
    }

    // Print the 8x8 grid of indices
    printf("Resulting array (flattened indices):\n");
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            printf("%3d ", h_data[y * WIDTH + x]);
        }
        printf("\n");
    }

    // Clean up
    cudaFree(d_data);
    free(h_data);

    return EXIT_SUCCESS;
}
```