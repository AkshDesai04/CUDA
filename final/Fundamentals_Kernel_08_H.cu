/*
Aim of the program:
Define the grid dimensions using a `dim3` variable: `dim3 grid_dim(4, 4);`.

Thinking:
The task is to create a minimal CUDA C program that demonstrates how to set grid dimensions
with a `dim3` variable. To make the example meaningful, the program will launch a simple
kernel that writes the global thread index into an array. The host code will allocate memory
on the device, launch the kernel with the specified grid dimensions, copy the results back
to the host, and print them. This illustrates both the usage of `dim3 grid_dim(4, 4);`
and basic CUDA memory management and kernel launch syntax.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that writes the global thread index into an output array
__global__ void write_thread_idx(int *out)
{
    // Compute global thread index
    int gid = blockIdx.x * blockDim.x + threadIdx.x
            + blockIdx.y * blockDim.y * gridDim.x;
    // Store the index in the output array
    out[gid] = gid;
}

int main(void)
{
    // Define block size (e.g., 8x8 threads per block)
    const dim3 block_dim(8, 8);
    // Define grid size as specified in the prompt
    dim3 grid_dim(4, 4);

    // Total number of threads
    const int total_threads = grid_dim.x * grid_dim.y * block_dim.x * block_dim.y;

    // Allocate host memory
    int *h_out = (int*)malloc(total_threads * sizeof(int));
    if (h_out == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    int *d_out = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_out, total_threads * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        free(h_out);
        return EXIT_FAILURE;
    }

    // Launch kernel
    write_thread_idx<<<grid_dim, block_dim>>>(d_out);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return EXIT_FAILURE;
    }

    // Copy results back to host
    err = cudaMemcpy(h_out, d_out, total_threads * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return EXIT_FAILURE;
    }

    // Print results
    printf("Global thread indices written by the kernel:\n");
    for (int y = 0; y < grid_dim.y; ++y) {
        for (int x = 0; x < grid_dim.x; ++x) {
            for (int j = 0; j < block_dim.y; ++j) {
                for (int i = 0; i < block_dim.x; ++i) {
                    int gid = ((y * grid_dim.x + x) * block_dim.x * block_dim.y)
                            + (j * block_dim.x + i);
                    printf("%5d ", h_out[gid]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    // Clean up
    cudaFree(d_out);
    free(h_out);

    return EXIT_SUCCESS;
}
