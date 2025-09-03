```cuda
/* 
Aim of the program: Define the block dimensions using a `dim3` variable: `dim3 block_dim(8, 8);` 
This program demonstrates how to set block dimensions using a dim3 variable in CUDA C++. It allocates a
small array on the host, copies it to the device, launches a simple kernel that writes the thread
indices into the array, and then copies the result back to the host and prints it. The block
dimensions are explicitly set to (8,8) as requested. The grid dimensions are computed to cover
the entire array. The kernel writes the linear thread index to the corresponding element in the
array. This example illustrates how block dimensions affect kernel launch configuration and
thread indexing. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void writeThreadIdx(int *d_out, int width, int height)
{
    // Compute 2D thread coordinates
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx < width && ty < height) {
        int idx = ty * width + tx;
        d_out[idx] = ty * width + tx;  // store linear index
    }
}

int main()
{
    const int width = 16;   // example width
    const int height = 12;  // example height
    const int size = width * height;
    const int bytes = size * sizeof(int);

    // Allocate host memory
    int *h_array = (int*)malloc(bytes);
    if (!h_array) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    int *d_array;
    cudaError_t err = cudaMalloc((void**)&d_array, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        free(h_array);
        return EXIT_FAILURE;
    }

    // Define block dimensions as requested
    dim3 block_dim(8, 8);  // 8x8 threads per block

    // Compute grid dimensions to cover the entire array
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x,
                  (height + block_dim.y - 1) / block_dim.y);

    // Launch kernel
    writeThreadIdx<<<grid_dim, block_dim>>>(d_array, width, height);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_array);
        free(h_array);
        return EXIT_FAILURE;
    }

    // Copy result back to host
    err = cudaMemcpy(h_array, d_array, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Memcpy device to host failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_array);
        free(h_array);
        return EXIT_FAILURE;
    }

    // Print results
    printf("Result array (linear indices written by kernel):\n");
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printf("%4d ", h_array[y * width + x]);
        }
        printf("\n");
    }

    // Clean up
    cudaFree(d_array);
    free(h_array);

    return EXIT_SUCCESS;
}
```