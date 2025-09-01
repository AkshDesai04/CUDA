/*
Aim: Launch with a 4x4 grid of 8x8 blocks. Each thread should write its global flattened ID to a 2D output matrix.

Thinking:
- A 4x4 grid of 8x8 blocks results in 32x32 threads in total.
- The global 2D coordinates of a thread are computed as:
    global_x = blockIdx.x * blockDim.x + threadIdx.x
    global_y = blockIdx.y * blockDim.y + threadIdx.y
- The flattened global ID in row-major order is:
    idx = global_y * total_cols + global_x
  where total_cols = gridDim.x * blockDim.x.
- Each thread will write this idx into a 2D output matrix stored in a linear array on the device.
- The host will allocate a matching 32x32 integer array, launch the kernel, copy the result back, and print it row by row.
- Basic CUDA error checking is included for allocation, kernel launch, and memory copy.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_X 8
#define BLOCK_Y 8
#define GRID_X  4
#define GRID_Y  4

__global__ void write_global_ids(int *d_out, int total_cols, int total_rows)
{
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure we are within bounds (optional here, as grid and block match array size)
    if (global_x < total_cols && global_y < total_rows)
    {
        int idx = global_y * total_cols + global_x;
        d_out[idx] = idx;  // store flattened global ID
    }
}

int main(void)
{
    const int total_cols = GRID_X * BLOCK_X; // 32
    const int total_rows = GRID_Y * BLOCK_Y; // 32
    const int total_elements = total_rows * total_cols;

    int *h_out = (int *)malloc(total_elements * sizeof(int));
    if (h_out == NULL)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return 1;
    }

    int *d_out;
    cudaError_t err = cudaMalloc((void **)&d_out, total_elements * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_out);
        return 1;
    }

    dim3 blockDim(BLOCK_X, BLOCK_Y);
    dim3 gridDim(GRID_X, GRID_Y);

    write_global_ids<<<gridDim, blockDim>>>(d_out, total_cols, total_rows);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return 1;
    }

    err = cudaMemcpy(h_out, d_out, total_elements * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return 1;
    }

    // Print the matrix
    printf("2D Output Matrix (each value is its flattened global ID):\n");
    for (int y = 0; y < total_rows; ++y)
    {
        for (int x = 0; x < total_cols; ++x)
        {
            printf("%4d ", h_out[y * total_cols + x]);
        }
        printf("\n");
    }

    cudaFree(d_out);
    free(h_out);
    return 0;
}
