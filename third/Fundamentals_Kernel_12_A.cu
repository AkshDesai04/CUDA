```cpp
/* 
Set up a 1D problem. Define `int N = 1024; int threads_per_block = 256;`. 
Calculate the number of blocks needed: `int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;`. 
Create `dim3` variables from this and launch.

Thinking:
- The goal is to demonstrate a minimal CUDA program that sets up a 1D problem space.
- We need to define the problem size `N` and the number of threads per block.
- The number of blocks per grid is calculated with the usual ceiling division formula.
- Using `dim3` objects to specify the grid and block dimensions is required.
- A simple kernel that writes the global index into an array will illustrate the launch.
- After launching, we copy the data back to host and print a few values to confirm.
- Error checking is added to ensure robustness.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Simple kernel that writes its global index into an array
__global__ void set_values(int* d_array, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        d_array[idx] = idx;
    }
}

int main()
{
    const int N = 1024;
    const int threads_per_block = 256;
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // Define dim3 variables
    dim3 blockSize(threads_per_block);
    dim3 gridSize(blocks_per_grid);

    // Allocate device memory
    int* d_array;
    cudaError_t err = cudaMalloc((void**)&d_array, N * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory (error code %d)\n", err);
        return EXIT_FAILURE;
    }

    // Launch kernel
    set_values<<<gridSize, blockSize>>>(d_array, N);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed (error code %d)\n", err);
        cudaFree(d_array);
        return EXIT_FAILURE;
    }

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device synchronization failed (error code %d)\n", err);
        cudaFree(d_array);
        return EXIT_FAILURE;
    }

    // Allocate host memory and copy results
    int h_array[N];
    err = cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed (error code %d)\n", err);
        cudaFree(d_array);
        return EXIT_FAILURE;
    }

    // Print first 10 values to verify
    printf("First 10 values of the array:\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_array);
    return EXIT_SUCCESS;
}
```