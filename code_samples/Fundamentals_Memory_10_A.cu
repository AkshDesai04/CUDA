/*
Write a kernel that accesses the 2D array on the device. To access element (x, y), a thread must use the pitch: 
`T* row = (T*)((char*)d_ptr + y * pitch); T element = row[x];`.

Thoughts:
- The program will allocate a 2D array on the host, fill it with some values, and copy it to the device using cudaMallocPitch to obtain a pitched allocation.
- The kernel will be templated for int type for simplicity; in practice this could be a macro or generic template in C++.
- Each thread will compute its (x, y) coordinates from blockIdx, blockDim, and threadIdx.
- Using the provided pitch-based row access, the kernel will read the element at (x, y), multiply it by 2, and write it to an output array.
- The output array will also be allocated with cudaMallocPitch to preserve the same memory layout.
- After kernel execution, the host will copy back the output array, print a few values to verify correct operation, and free all resources.
- Error checking is added via a helper macro to keep the code readable.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Kernel that accesses a 2D pitched array
__global__ void processPitchedArray(int *d_in, size_t pitch_in,
                                    int *d_out, size_t pitch_out,
                                    int width, int height)
{
    // Compute 2D coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds check
    if (x >= width || y >= height) return;

    // Access input element using pitch
    int *row_in = (int*)((char*)d_in + y * pitch_in);
    int element = row_in[x];

    // Process (here, just multiply by 2)
    int result = element * 2;

    // Write to output using pitch
    int *row_out = (int*)((char*)d_out + y * pitch_out);
    row_out[x] = result;
}

int main(void)
{
    // Define dimensions
    const int width  = 8;
    const int height = 6;

    // Host allocation (row-major)
    int *h_in  = (int*)malloc(width * height * sizeof(int));
    int *h_out = (int*)malloc(width * height * sizeof(int));

    // Initialize host input
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            h_in[y * width + x] = y * width + x; // simple pattern
        }
    }

    // Device pitched allocation
    int *d_in;
    size_t pitch_in;
    CUDA_CHECK(cudaMallocPitch((void**)&d_in, &pitch_in,
                               width * sizeof(int), height));

    int *d_out;
    size_t pitch_out;
    CUDA_CHECK(cudaMallocPitch((void**)&d_out, &pitch_out,
                               width * sizeof(int), height));

    // Copy host to device
    CUDA_CHECK(cudaMemcpy2D(d_in, pitch_in,
                            h_in, width * sizeof(int),
                            width * sizeof(int), height,
                            cudaMemcpyHostToDevice));

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width  + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    processPitchedArray<<<gridSize, blockSize>>>(d_in, pitch_in,
                                                 d_out, pitch_out,
                                                 width, height);
    CUDA_CHECK(cudaGetLastError());

    // Copy back result
    CUDA_CHECK(cudaMemcpy2D(h_out, width * sizeof(int),
                            d_out, pitch_out,
                            width * sizeof(int), height,
                            cudaMemcpyDeviceToHost));

    // Print some results
    printf("Input array:\n");
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printf("%4d", h_in[y * width + x]);
        }
        printf("\n");
    }

    printf("\nOutput array (input * 2):\n");
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printf("%4d", h_out[y * width + x]);
        }
        printf("\n");
    }

    // Clean up
    free(h_in);
    free(h_out);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
