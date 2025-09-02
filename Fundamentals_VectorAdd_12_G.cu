/*
Launch with a 1D grid of 1D blocks instead. Each thread must manually compute its (x,y) coordinate from its global 1D ID: int x = id % width; int y = id / width;

Thinking:
The goal is to demonstrate how to launch a kernel with a one-dimensional grid of one-dimensional blocks, yet still map threads to two-dimensional data by manually computing x and y coordinates from the global thread ID.  
We will allocate a 2D array on the host (flattened into a 1D array), copy it to the device, launch a kernel that processes each element, and copy the result back.  
The kernel will compute the global thread ID as `int id = blockIdx.x * blockDim.x + threadIdx.x;`.  
It will then calculate the 2D coordinates:
```
int x = id % width;
int y = id / width;
```
Using these coordinates we can index into the flattened array as `int idx = y * width + x;`.  
The processing done in this example will simply store the sum of the coordinates (`x + y`) into an output array.  
This illustrates how the mapping works and ensures that the kernel correctly handles cases where the total number of threads may exceed the array size (those threads will exit early).  
Error checking for CUDA calls is included for robustness.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",           \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

__global__ void computeKernel(const float* input, float* output,
                               int width, int height)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (id >= total) return;

    int x = id % width;
    int y = id / width;
    int idx = y * width + x;

    // Example operation: store the sum of coordinates
    output[idx] = input[idx] + x + y;
}

int main(void)
{
    const int width = 8;
    const int height = 6;
    const int size = width * height;
    const int bytes = size * sizeof(float);

    // Allocate host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);

    // Initialize input with some values
    for (int i = 0; i < size; ++i) {
        h_input[i] = (float)i;
    }

    // Allocate device memory
    float *d_input = NULL;
    float *d_output = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_input, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, bytes));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Define block and grid sizes
    const int blockSize = 256;
    const int gridSize = (size + blockSize - 1) / blockSize;

    // Launch kernel
    computeKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    // Print some results
    printf("Results (first 20 elements):\n");
    for (int i = 0; i < 20 && i < size; ++i) {
        printf("%.1f ", h_output[i]);
    }
    printf("\n");

    // Clean up
    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
