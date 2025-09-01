/*
Aim: Combine the concepts of the previous two questions. Launch a kernel with a grid of 4 blocks, each containing 16 threads. Each thread must calculate and print its unique 'global ID' using the formula: `int id = blockIdx.x * blockDim.x + threadIdx.x;`.

Thinking:
To satisfy this requirement, we need to write a simple CUDA program that does the following:
1. Define a kernel that is executed by many threads.  
2. Inside the kernel each thread computes its global index by using the built‑in CUDA variables `blockIdx.x`, `blockDim.x`, and `threadIdx.x`.  
3. Each thread then prints its global ID using `printf` from the device.  CUDA supports `printf` from device code, but we must include the appropriate header and compile with the `-arch` flag to enable the feature.  
4. The host program sets up a grid of 4 blocks (`gridDim.x = 4`) and a block size of 16 threads (`blockDim.x = 16`).  
5. We launch the kernel with these dimensions, then optionally call `cudaDeviceReset()` or `cudaDeviceSynchronize()` to ensure all prints are flushed before the program exits.  
6. For safety, we can check the return status of the kernel launch with `cudaGetLastError()`.

The resulting program is straightforward and demonstrates basic kernel launch, thread indexing, and device printing.

*/

#include <stdio.h>
#include <cuda.h>

// Kernel that prints the global ID of each thread
__global__ void print_global_id()
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread (block %d, thread %d) has global ID: %d\n",
           blockIdx.x, threadIdx.x, id);
}

int main()
{
    // Define grid and block dimensions
    dim3 gridDim(4);   // 4 blocks in the grid
    dim3 blockDim(16); // 16 threads per block

    // Launch the kernel
    print_global_id<<<gridDim, blockDim>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for the device to finish all tasks
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d, line(%d)\n", err, __LINE__);
        return 1;
    }

    // Clean up and reset device
    cudaDeviceReset();

    return 0;
}
