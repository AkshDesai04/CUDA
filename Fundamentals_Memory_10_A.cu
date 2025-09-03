/*
Aim: Write a kernel that accesses the 2D array on the device. To access element (x, y), a thread must use the pitch: `T* row = (T*)((char*)d_ptr + y * pitch); T element = row[x];`.

Thinking:
- I chose `int` as the element type (`T`) for simplicity, but the code can be templated if needed.
- The kernel will double each element in the array. Each thread corresponds to one element `(x, y)`.
- We allocate the 2D array on the host, initialize it with a simple pattern for easy verification.
- On the device, we use `cudaMallocPitch` to obtain a pitched allocation. Pitch is the width in bytes of the allocated memory, which may be larger than the logical row width due to alignment.
- `cudaMemcpy2D` is used to copy from the host to the device, and `cudaMemcpy2D` again to copy back after kernel execution.
- The kernel obtains the base pointer and pitch, casts the base pointer to `char*` so that pointer arithmetic can be performed in bytes, then adds `y * pitch` to get to the start of the correct row, and finally casts to `T*` to index the element by `x`.
- Bounds checking is performed in the kernel to avoid reading/writing out of bounds.
- After copying back, the program prints a few elements to demonstrate that the kernel worked correctly.
- The code includes necessary error checking macros for brevity and readability.
- Finally, resources are freed appropriately.

Note: This example is fully self‑contained and can be compiled with `nvcc -o pitch_example pitch_example.cu` and run on any system with CUDA installed.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Kernel that accesses a 2D array using pitch
__global__ void double_elements(int *d_ptr, size_t pitch, int width, int height)
{
    // Compute (x, y) coordinates of this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index

    // Bounds check
    if (x >= width || y >= height) return;

    // Cast base pointer to char* for byte-wise offset
    char *base = (char *)d_ptr;

    // Compute pointer to the start of row y
    int *row = (int *)(base + y * pitch);

    // Access the element at column x
    int element = row[x];

    // For demonstration, double the element
    row[x] = element * 2;
}

int main(void)
{
    // Dimensions of the 2D array
    const int width  = 8;   // number of columns
    const int height = 6;   // number of rows

    // Host allocation and initialization
    int h_array[height][width];
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            h_array[y][x] = y * width + x; // simple pattern

    // Device allocation with pitch
    int *d_ptr = nullptr;
    size_t pitch = 0;
    CUDA_CHECK(cudaMallocPitch((void **)&d_ptr, &pitch,
                               width * sizeof(int), height));

    // Copy host to device (2D copy)
    CUDA_CHECK(cudaMemcpy2D(d_ptr, pitch,
                            h_array, width * sizeof(int),
                            width * sizeof(int), height,
                            cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width  + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    double_elements<<<gridSize, blockSize>>>(d_ptr, pitch, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy2D(h_array, width * sizeof(int),
                            d_ptr, pitch,
                            width * sizeof(int), height,
                            cudaMemcpyDeviceToHost));

    // Print a few elements to verify
    printf("Resulting array after doubling:\n");
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x)
            printf("%4d ", h_array[y][x]);
        printf("\n");
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_ptr));

    return 0;
}
