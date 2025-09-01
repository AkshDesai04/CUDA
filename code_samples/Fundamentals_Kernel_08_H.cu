/*
Define the grid dimensions using a `dim3` variable: `dim3 grid_dim(4, 4);`.

Thinking:
1. The goal is to create a CUDA program that demonstrates how to define grid dimensions with a dim3 variable.
2. Use a simple kernel that writes thread indices to an array so that we can verify the grid and block configuration.
3. Set block dimensions to a small value (e.g., 2x2) so that with grid (4,4) we cover a reasonable problem size (8x8 threads).
4. Allocate device memory, launch kernel, copy results back, and print a subset of the output to confirm correctness.
5. Include minimal error checking for CUDA API calls.
6. Ensure the code is self-contained and compilable as a .cu file.

The program will:
- Define `dim3 block_dim(2, 2);` and `dim3 grid_dim(4, 4);`.
- Launch a kernel that computes a global linear index and stores it in an output array.
- Copy the array back to host and print the first few elements.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void compute_indices(int *out, int width, int height)
{
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    int grid_x   = blockIdx.x;
    int grid_y   = blockIdx.y;

    int block_w  = blockDim.x;
    int block_h  = blockDim.y;

    int global_x = grid_x * block_w + local_x;
    int global_y = grid_y * block_h + local_y;

    if (global_x < width && global_y < height) {
        int idx = global_y * width + global_x;
        out[idx] = idx;  // store the linear index
    }
}

int main(void)
{
    const int width  = 8;   // total width = grid_dim.x * block_dim.x
    const int height = 8;   // total height = grid_dim.y * block_dim.y

    const size_t size = width * height * sizeof(int);

    int *h_out = (int *)malloc(size);
    if (!h_out) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return 1;
    }

    int *d_out = NULL;
    cudaError_t err = cudaMalloc((void **)&d_out, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_out);
        return 1;
    }

    // Define block and grid dimensions
    dim3 block_dim(2, 2);
    dim3 grid_dim(4, 4);  // Aim: Define the grid dimensions using a `dim3` variable: `dim3 grid_dim(4, 4);`

    // Launch kernel
    compute_indices<<<grid_dim, block_dim>>>(d_out, width, height);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return 1;
    }

    // Print a portion of the output array
    printf("First 16 elements of the output array (linear indices):\n");
    for (int i = 0; i < 16; ++i) {
        printf("%4d ", h_out[i]);
        if ((i + 1) % 4 == 0) printf("\n");
    }

    // Clean up
    cudaFree(d_out);
    free(h_out);

    return 0;
}
