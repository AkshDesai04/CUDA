/*
Aim: Set up a 1D problem. Define `int N = 1024; int threads_per_block = 256;`. Calculate the number of blocks needed: `int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;`. Create `dim3` variables from this and launch.

Thinking:
- We need to compute the grid and block dimensions for a simple 1‑D CUDA kernel launch.
- N is the number of elements; threads_per_block is the number of threads in a block.
- The number of blocks is computed with the ceiling division formula to ensure all N elements are covered.
- We create `dim3 dimBlock(threads_per_block, 1, 1);` and `dim3 dimGrid(blocks_per_grid, 1, 1);`.
- For demonstration, we allocate an integer array on the device, launch a kernel that stores each thread’s global index into the array, copy the array back to the host, and print a few values to verify.
- Basic error checking on CUDA API calls is included for robustness.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel(int *d_arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        d_arr[idx] = idx;          // store global thread index
    }
}

int main(void)
{
    // Problem setup
    const int N = 1024;
    const int threads_per_block = 256;
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // Create dim3 variables for launch
    dim3 dimBlock(threads_per_block, 1, 1);
    dim3 dimGrid(blocks_per_grid, 1, 1);

    // Host and device memory
    int *h_arr = (int *)malloc(N * sizeof(int));
    if (!h_arr)
    {
        fprintf(stderr, "Host memory allocation failed\n");
        return 1;
    }

    int *d_arr = nullptr;
    cudaError_t err = cudaMalloc((void **)&d_arr, N * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_arr);
        return 1;
    }

    // Launch kernel
    kernel<<<dimGrid, dimBlock>>>(d_arr, N);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return 1;
    }

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return 1;
    }

    // Copy results back to host
    err = cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return 1;
    }

    // Print first 10 values to verify
    printf("First 10 elements of the array:\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("h_arr[%d] = %d\n", i, h_arr[i]);
    }

    // Cleanup
    cudaFree(d_arr);
    free(h_arr);

    return 0;
}
