/*
Aim: Choose a block size of 256 threads. Calculate the required grid size for a vector of 1,048,576 elements.

Thinking:
- The vector has 1,048,576 elements.
- We want to use 256 threads per block.
- The grid size (number of blocks) is computed as ceil(numElements / blockSize).
- Ceil division can be done as (numElements + blockSize - 1) / blockSize.
- For 1,048,576 / 256, the division is exact: 1,048,576 / 256 = 4,096.
- Thus the required grid size is 4,096 blocks.
- We'll implement a simple kernel that does nothing but will be launched with the computed grid and block dimensions to demonstrate usage.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void dummyKernel(float *d_array, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Simple operation (optional)
        d_array[idx] = d_array[idx] * 2.0f;
    }
}

int main() {
    const int numElements = 1048576;   // 1,048,576 elements
    const int blockSize   = 256;       // 256 threads per block

    // Calculate grid size
    int gridSize = (numElements + blockSize - 1) / blockSize;

    printf("Number of elements: %d\n", numElements);
    printf("Block size: %d\n", blockSize);
    printf("Grid size (number of blocks): %d\n", gridSize);

    // Allocate host memory
    float *h_array = (float *)malloc(numElements * sizeof(float));
    for (int i = 0; i < numElements; ++i) h_array[i] = static_cast<float>(i);

    // Allocate device memory
    float *d_array = nullptr;
    cudaMalloc((void **)&d_array, numElements * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_array, h_array, numElements * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dummyKernel<<<gridSize, blockSize>>>(d_array, numElements);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_array, d_array, numElements * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_array);
    free(h_array);

    printf("Kernel execution completed.\n");

    return 0;
}
