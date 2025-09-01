```cuda
/*
Aim: Launch with a 4x4 grid of 8x8 blocks. Each thread should write its global flattened ID to a 2D output matrix.

Thinking process:
1. Define the grid and block dimensions: gridDim.x = 4, gridDim.y = 4; blockDim.x = 8, blockDim.y = 8.
2. The total number of threads in x direction = gridDim.x * blockDim.x = 32, similarly for y = 32.
3. Each thread will compute its global (flattened) ID by treating the entire thread space as a 1D array laid out in row-major order:
   globalIdx = y * width + x
   where width = total threads in x direction, and y is the global y index, x is the global x index.
4. The kernel writes this globalIdx into a 2D output array that we represent as a 1D array with row-major indexing.
5. On the host, we allocate a matching 2D array, launch the kernel, copy the data back, and print it.
6. Include basic CUDA error checking.
*/

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            std::cerr << "CUDA error in " << __FILE__ << ":"       \
                      << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel that writes the global flattened thread ID to a 2D output matrix
__global__ void writeThreadIDs(int* output, int width, int height)
{
    // Compute global thread indices
    int x = threadIdx.x + blockIdx.x * blockDim.x; // global x
    int y = threadIdx.y + blockIdx.y * blockDim.y; // global y

    // Bounds check (should not be necessary with full launch bounds, but safe)
    if (x < width && y < height)
    {
        int globalIdx = y * width + x;
        output[y * width + x] = globalIdx;
    }
}

int main()
{
    // Define grid and block dimensions
    const int gridDimX = 4;
    const int gridDimY = 4;
    const int blockDimX = 8;
    const int blockDimY = 8;

    // Compute total dimensions of the output matrix
    const int width  = gridDimX * blockDimX; // 32
    const int height = gridDimY * blockDimY; // 32

    const size_t totalSize = static_cast<size_t>(width) * height;

    // Allocate host memory
    int* h_output = new int[totalSize];

    // Allocate device memory
    int* d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_output, totalSize * sizeof(int)));

    // Launch kernel
    dim3 block(blockDimX, blockDimY);
    dim3 grid(gridDimX, gridDimY);
    writeThreadIDs<<<grid, block>>>(d_output, width, height);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure kernel completion

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, totalSize * sizeof(int), cudaMemcpyDeviceToHost));

    // Print the 2D output matrix
    std::cout << "2D output matrix (" << height << "x" << width << "):" << std::endl;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            std::cout << std::setw(4) << h_output[y * width + x];
        }
        std::cout << std::endl;
    }

    // Clean up
    delete[] h_output;
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
```