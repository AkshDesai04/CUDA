/*
Set up a 1D problem. Define `int N = 1024; int threads_per_block = 256;`. Calculate the number of blocks needed: `int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;`. Create `dim3` variables from this and launch.

Thinking:
- We need a minimal CUDA program that demonstrates setting up the grid and block dimensions for a 1D kernel launch.
- Define the problem size N=1024 and the number of threads per block as 256.
- Compute the number of blocks required using the ceiling division formula.
- Create dim3 variables for both grid and block dimensions to pass to the kernel launch.
- Provide a simple kernel (e.g., initialize an array with its index or double its value) to confirm the launch works.
- Allocate host and device memory, copy data, launch the kernel, copy back results, and verify.
- Include error checking for CUDA API calls to make the program robust.
- Ensure the program is self-contained and can compile with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that writes the global index to the output array
__global__ void init_kernel(int *d_out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        d_out[idx] = idx;  // For example, store the index
}

int main()
{
    const int N = 1024;
    const int threads_per_block = 256;
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // Define dim3 variables
    dim3 blockDim(threads_per_block);
    dim3 gridDim(blocks_per_grid);

    printf("Launching kernel with %d blocks of %d threads each (total %d threads)\n",
           blocks_per_grid, threads_per_block, blocks_per_grid * threads_per_block);

    // Allocate host memory
    int *h_out = (int*)malloc(N * sizeof(int));
    if (h_out == NULL)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    int *d_out;
    cudaError_t err = cudaMalloc((void**)&d_out, N * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_out);
        return EXIT_FAILURE;
    }

    // Launch kernel
    init_kernel<<<gridDim, blockDim>>>(d_out, N);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return EXIT_FAILURE;
    }

    // Copy results back to host
    err = cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy (device to host) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return EXIT_FAILURE;
    }

    // Verify results
    int success = 1;
    for (int i = 0; i < N; ++i)
    {
        if (h_out[i] != i)
        {
            printf("Mismatch at index %d: expected %d, got %d\n", i, i, h_out[i]);
            success = 0;
            break;
        }
    }
    if (success)
        printf("All values correct.\n");

    // Clean up
    cudaFree(d_out);
    free(h_out);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
