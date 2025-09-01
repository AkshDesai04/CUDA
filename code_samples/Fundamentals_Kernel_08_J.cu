```cuda
/*
Aim of the program:
Launch with a 4x4 grid of 8x8 blocks. Each thread should write its global flattened ID to a 2D output matrix.

Thinking and approach:
- The grid consists of 4 blocks in the x‑dimension and 4 blocks in the y‑dimension.  
- Each block is 8 threads wide (x) and 8 threads tall (y).  
- Total threads = 4*4*8*8 = 1024.  
- The program will allocate a linear array on the device that represents a 2‑D matrix of size
  width = gridDim.x * blockDim.x = 32 and height = gridDim.y * blockDim.y = 32.
- Each thread computes its global coordinates:
    globalX = threadIdx.x + blockIdx.x * blockDim.x
    globalY = threadIdx.y + blockIdx.y * blockDim.y
  The linear index into the output array is:
    idx = globalX + globalY * width
- The flattened global thread ID can be computed as:
    globalId = blockIdx.x * blockDim.x * blockDim.y * gridDim.y
             + blockIdx.y * blockDim.x * blockDim.y
             + threadIdx.y * blockDim.x
             + threadIdx.x
- The kernel writes globalId into output[idx].
- After the kernel finishes, the host copies the array back and prints it as a 32x32 matrix.
- Basic error checking for CUDA API calls is included for robustness.

The program is self‑contained, compiles with nvcc, and demonstrates how to map a 2‑D grid of blocks and threads to a 2‑D output matrix.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Utility macro for checking CUDA API calls */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

/* Kernel: Each thread writes its global flattened ID to a 2D output matrix */
__global__ void writeGlobalId(int *output, int width, int height)
{
    // Compute global coordinates
    int globalX = threadIdx.x + blockIdx.x * blockDim.x;
    int globalY = threadIdx.y + blockIdx.y * blockDim.y;

    // Compute linear index into the output array
    int idx = globalX + globalY * width;

    // Compute flattened global thread ID
    int globalId = blockIdx.x * blockDim.x * blockDim.y * gridDim.y
                 + blockIdx.y * blockDim.x * blockDim.y
                 + threadIdx.y * blockDim.x
                 + threadIdx.x;

    // Write to the output matrix
    if (globalX < width && globalY < height) {
        output[idx] = globalId;
    }
}

int main(void)
{
    // Define grid and block dimensions
    dim3 blockDim(8, 8);   // 8x8 threads per block
    dim3 gridDim(4, 4);    // 4x4 blocks

    // Compute matrix dimensions
    const int width  = gridDim.x * blockDim.x; // 32
    const int height = gridDim.y * blockDim.y; // 32
    const size_t numElements = (size_t)width * (size_t)height;

    // Allocate device memory
    int *d_output = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_output, numElements * sizeof(int)));

    // Launch kernel
    writeGlobalId<<<gridDim, blockDim>>>(d_output, width, height);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

    // Allocate host memory and copy result back
    int *h_output = (int *)malloc(numElements * sizeof(int));
    if (!h_output) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        CUDA_CHECK(cudaFree(d_output));
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaMemcpy(h_output, d_output, numElements * sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Print the 2D matrix
    printf("Output matrix (%dx%d):\n", height, width);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printf("%4d ", h_output[y * width + x]);
        }
        printf("\n");
    }

    // Clean up
    free(h_output);
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
```