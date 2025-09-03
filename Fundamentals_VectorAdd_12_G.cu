/*
Aim:
Launch with a 1D grid of 1D blocks instead. Each thread must manually compute its (x,y) coordinate from its global 1D ID: int x = id % width; int y = id / width.

Thinking:
The requirement is to demonstrate a CUDA program that uses a one‑dimensional grid and one‑dimensional blocks, and for every thread it computes its 2‑D coordinates manually from the global thread index.  
To keep the example complete and runnable, the kernel will write a simple value (x + y) into a linear output array that represents a 2‑D image of size `width × height`.  
The host code allocates the host and device arrays, launches the kernel with appropriate grid/block dimensions, copies the result back, and prints a few samples to verify correctness.  
Error checking is added via a small macro.  The code compiles as a standard `.cu` file and can be built with `nvcc`.

*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

__global__ void computeKernel(int width, int height, int *output)
{
    int id   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;

    if (id < total)
    {
        int x = id % width;   // manual computation of x
        int y = id / width;   // manual computation of y

        // Example operation: store the sum of coordinates
        output[id] = x + y;
    }
}

int main(void)
{
    const int width  = 256;   // image width
    const int height = 256;   // image height
    const int total  = width * height;

    // Host memory
    int *h_output = (int *)malloc(total * sizeof(int));
    if (!h_output) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Device memory
    int *d_output = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_output, total * sizeof(int)));

    // Kernel launch parameters
    const int blockSize = 256;                                 // 1D block
    const int gridSize  = (total + blockSize - 1) / blockSize; // 1D grid

    // Launch kernel
    computeKernel<<<gridSize, blockSize>>>(width, height, d_output);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, total * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify some results
    printf("Sample outputs (x, y, x+y):\n");
    for (int y = 0; y < 5; ++y)
    {
        for (int x = 0; x < 5; ++x)
        {
            int idx = y * width + x;
            printf("(%d, %d) -> %d  ", x, y, h_output[idx]);
        }
        printf("\n");
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_output));
    free(h_output);

    return 0;
}
