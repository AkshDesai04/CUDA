```cuda
/*
Aim of the program:
Write a kernel that accesses the 2D array on the device. To access element (x, y), a thread must use the pitch:
T* row = (T*)((char*)d_ptr + y * pitch);
T element = row[x];

Thinking:
The program demonstrates how to allocate a 2D array on the GPU using cudaMallocPitch, launch a kernel that
accesses each element using the pitch (to correctly index the memory due to possible padding), perform a simple
operation (increment each element by 1.0f), copy the result back to the host, and verify the operation.
Key points:
1. cudaMallocPitch returns a pointer `d_ptr` and a pitch value in bytes.
2. Inside the kernel, cast `d_ptr` to `char*` then add `y * pitch` to get the start of the y-th row.
3. Cast that to the element type `T*` to index the x-th element.
4. Use a 2D grid of blocks and threads to cover the entire array.
5. Error checking macros are provided for readability.
6. The host code initializes a 2D array, copies it to the device, runs the kernel, copies back, and prints a few elements to confirm.
*/

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call)                                                    \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",            \
                    __FILE__, __LINE__, err, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

// Kernel that increments each element of a 2D float array using pitch
__global__ void incrementKernel(float* d_ptr, size_t pitch, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // column index
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // row index

    if (x < width && y < height)
    {
        // Compute pointer to the start of row y
        float* row = (float*)((char*)d_ptr + y * pitch);
        // Access element (x, y) and increment
        row[x] += 1.0f;
    }
}

int main(void)
{
    const int width = 10;   // number of columns
    const int height = 5;   // number of rows
    const size_t elemSize = sizeof(float);

    // Allocate host memory
    float* h_array = (float*)malloc(width * height * elemSize);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            h_array[y * width + x] = (float)(y * width + x);  // Initialize with unique values

    // Allocate device memory with pitch
    float* d_ptr;
    size_t pitch;  // in bytes
    CHECK_CUDA(cudaMallocPitch(&d_ptr, &pitch, width * elemSize, height));

    // Copy host array to device
    CHECK_CUDA(cudaMemcpy2D(d_ptr, pitch,
                            h_array, width * elemSize,
                            width * elemSize, height,
                            cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    incrementKernel<<<gridDim, blockDim>>>(d_ptr, pitch, width, height);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy2D(h_array, width * elemSize,
                            d_ptr, pitch,
                            width * elemSize, height,
                            cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("First 5 elements after increment:\n");
    for (int i = 0; i < 5; ++i)
        printf("%f ", h_array[i]);
    printf("\n");

    // Clean up
    free(h_array);
    CHECK_CUDA(cudaFree(d_ptr));

    return 0;
}
```