```cpp
/*
Aim of the program: 
Launch an 8x8 block. Pass a 2D array (flattened to 1D) `int* data`. Have each thread write its flattened 1D index to the corresponding location in the array.

Thinking:
- The kernel needs to be launched with a block of size 8x8 (i.e., dim3 blockDim(8,8,1)).
- Since we only have one block, gridDim can be (1,1,1).
- The flattened index for a 2D array stored in row-major order is computed as: idx = threadIdx.y * blockDim.x + threadIdx.x.
- Each thread writes its idx into data[idx].
- On the host side we allocate an array of size 64 (8*8) on device and on host.
- After kernel launch, copy the array back to host and optionally print the values to verify correctness.
- Include basic CUDA error checking for robustness.
- Ensure the code compiles as a .cu file with proper includes and using CUDA runtime API.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void writeFlattenedIndex(int *data, int width)
{
    // Compute 2D coordinates of the thread
    int x = threadIdx.x;
    int y = threadIdx.y;
    // Flattened index assuming row-major order
    int idx = y * width + x;
    // Write the flattened index to the array
    data[idx] = idx;
}

int checkCudaError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: %s : %s\n", msg, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int main()
{
    const int width = 8;
    const int height = 8;
    const int size = width * height * sizeof(int);

    // Allocate host memory
    int *h_data = (int *)malloc(size);
    if (!h_data)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return 1;
    }

    // Allocate device memory
    int *d_data;
    cudaMalloc((void **)&d_data, size);
    if (checkCudaError("cudaMalloc")) return 1;

    // Define block and grid dimensions
    dim3 blockDim(8, 8, 1);
    dim3 gridDim(1, 1, 1);

    // Launch kernel
    writeFlattenedIndex<<<gridDim, blockDim>>>(d_data, width);
    if (checkCudaError("Kernel launch")) return 1;

    // Copy result back to host
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    if (checkCudaError("cudaMemcpy DeviceToHost")) return 1;

    // Print the array to verify
    printf("Resulting flattened indices:\n");
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            printf("%2d ", h_data[y * width + x]);
        }
        printf("\n");
    }

    // Clean up
    cudaFree(d_data);
    free(h_data);

    return 0;
}
```