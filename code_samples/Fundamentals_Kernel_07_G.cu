```cpp
/*
Aim: Launch an 8x8 block. Pass a 2D array (flattened to 1D) `int* data`. Have each thread write its flattened 1D index to the corresponding location in the array.

Thinking:
1. Define a kernel that takes an int* array and writes to its element based on the thread's linear index.
2. Since we have a single 8x8 block, threadIdx.x ranges 0-7 and threadIdx.y ranges 0-7. The flattened index is threadIdx.y * blockDim.x + threadIdx.x.
3. The host will allocate an array of 64 ints on both host and device, launch the kernel with <<<1, dim3(8,8)>>>, copy back, and print the result to verify.
4. Add basic error checking for CUDA calls and simple output formatting.

*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void writeIndices(int* data, int width, int height) {
    // Compute the linear index for this thread within the block
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        data[idx] = idx;
    }
}

int main() {
    const int width = 8;
    const int height = 8;
    const int size = width * height * sizeof(int);

    // Allocate host memory
    int* h_data = (int*)malloc(size);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return 1;
    }

    // Allocate device memory
    int* d_data;
    cudaError_t err = cudaMalloc((void**)&d_data, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_data);
        return 1;
    }

    // Launch kernel
    dim3 threadsPerBlock(8, 8);
    dim3 blocksPerGrid(1);
    writeIndices<<<blocksPerGrid, threadsPerBlock>>>(d_data, width, height);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return 1;
    }

    // Print the array
    printf("Resulting array (flattened 8x8):\n");
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printf("%3d ", h_data[y * width + x]);
        }
        printf("\n");
    }

    // Clean up
    cudaFree(d_data);
    free(h_data);
    return 0;
}
```