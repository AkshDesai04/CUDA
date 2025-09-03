/*
Aim of the program:
In the loop, thread_id is the global index blockIdx.x * blockDim.x + threadIdx.x. grid_size is the total number of threads launched gridDim.x * blockDim.x. Implement this.

Thinking:
The task requires us to demonstrate how to compute a thread's global index (thread_id) and the total number of threads in a CUDA kernel launch (grid_size). 
To make this concrete we will:
1. Define a kernel that each thread prints its global index and the total number of launched threads.
2. The kernel will also write a simple value (here: the thread_id) into a device array, showing that the global index can be used for data indexing.
3. On the host side we allocate and initialize the data, launch the kernel with a configurable number of threads per block, copy back the result, and display it.
4. Using `printf` inside the kernel gives a clear, real-time trace of `thread_id` and `grid_size`, but this requires a compute capability >= 2.0.
5. The host program also prints the grid and block dimensions for verification.

The program is self‑contained in a single .cu file and can be compiled with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that demonstrates thread_id and grid_size
__global__ void demoKernel(int *output)
{
    // Compute global thread index
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // Compute total number of threads launched
    int grid_size = gridDim.x * blockDim.x;

    // Each thread writes its id to the output array if within bounds
    if (thread_id < grid_size) {
        output[thread_id] = thread_id;
        // Print the thread id and total grid size (for debugging/illustration)
        printf("Thread %d / %d: grid_size = %d\n", thread_id, grid_size, grid_size);
    }
}

int main(void)
{
    // Define grid and block dimensions
    const int threadsPerBlock = 8; // Example: 8 threads per block
    const int numberOfBlocks = 4;  // Example: 4 blocks
    const int grid_size = threadsPerBlock * numberOfBlocks; // total threads

    // Allocate host memory
    int *h_output = (int *)malloc(grid_size * sizeof(int));
    if (h_output == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return 1;
    }

    // Allocate device memory
    int *d_output;
    cudaError_t err = cudaMalloc((void **)&d_output, grid_size * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_output);
        return 1;
    }

    // Launch the kernel
    demoKernel<<<numberOfBlocks, threadsPerBlock>>>(d_output);

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_output);
        free(h_output);
        return 1;
    }

    // Copy results back to host
    err = cudaMemcpy(h_output, d_output, grid_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_output);
        free(h_output);
        return 1;
    }

    // Print the results
    printf("\n--- Result Array ---\n");
    for (int i = 0; i < grid_size; ++i) {
        printf("h_output[%d] = %d\n", i, h_output[i]);
    }

    // Cleanup
    cudaFree(d_output);
    free(h_output);

    // Reset device (optional)
    cudaDeviceReset();

    return 0;
}
