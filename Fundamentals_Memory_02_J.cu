```cuda
/*
Aim of the program: Write a function `void print_device_array(int* d_array, int size)` that allocates temporary host memory, copies from device, prints, and then frees the temporary memory.

Thinking:
1. We need to include the CUDA runtime header `<cuda_runtime.h>` and standard IO header `<stdio.h>`.
2. The function receives a device pointer `d_array` and its length `size`.
3. Allocate host memory using `malloc` (or `new`) for an array of `int` of length `size`.
4. Use `cudaMemcpy` to copy data from device to host. The direction is `cudaMemcpyDeviceToHost`.
   - We should check the return value of `cudaMemcpy` for errors and handle them gracefully.
5. Print the array elements to `stdout`. For readability, print elements separated by spaces and wrap with newlines.
6. After printing, free the host memory with `free`.
7. Optionally, provide a small `main` that demonstrates usage: allocate a device array, fill it with some values via a simple kernel, then call `print_device_array`.
8. Ensure the code compiles as a CUDA source file (`.cu`) and can be built with `nvcc`.

Below is the complete .cu file implementing this logic.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple kernel to initialize device array */
__global__ void init_array_kernel(int *d_arr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_arr[idx] = idx * 2;   // example initialization
}

/* Function to copy device array to host, print, and free host memory */
void print_device_array(int* d_array, int size)
{
    // Allocate temporary host memory
    int *h_arr = (int*)malloc(size * sizeof(int));
    if (h_arr == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return;
    }

    // Copy from device to host
    cudaError_t err = cudaMemcpy(h_arr, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(h_arr);
        return;
    }

    // Print the array
    printf("Device array contents (size=%d):\n", size);
    for (int i = 0; i < size; ++i) {
        printf("%d ", h_arr[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }
    if (size % 16 != 0) printf("\n");

    // Free temporary host memory
    free(h_arr);
}

int main(void)
{
    const int size = 64;
    int *d_arr = NULL;
    size_t bytes = size * sizeof(int);

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_arr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernel to initialize array
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    init_array_kernel<<<blocks, threadsPerBlock>>>(d_arr, size);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    // Wait for GPU to finish before accessing data
    cudaDeviceSynchronize();

    // Print device array
    print_device_array(d_arr, size);

    // Clean up
    cudaFree(d_arr);
    return 0;
}
```