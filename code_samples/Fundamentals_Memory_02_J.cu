/*
Write a function `void print_device_array(int* d_array, int size)` that allocates temporary host memory, copies from device, prints, and then frees the temporary memory.

Thinking process:
1. We need to allocate a temporary host buffer of the same size as the device array.
2. Use cudaMemcpy with cudaMemcpyDeviceToHost to copy data from device to host.
3. Iterate over the host array and print each element. For clarity, we will print them in a single line separated by spaces, and add a newline at the end.
4. After printing, free the allocated host memory with free().
5. Include error checking for the cudaMemcpy call so that any failure is reported.
6. To make the code a complete, compilable .cu file, we will include necessary headers (`stdio.h` for printing and `cuda_runtime.h` for CUDA API). 
7. For demonstration, a `main` function will allocate an array on the device, initialize it with some values, call `print_device_array`, and then clean up the device memory.
8. The code follows CUDA C++ conventions but uses plain C where possible, as requested.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Function that copies a device array to host, prints it, and frees temporary host memory
void print_device_array(int* d_array, int size) {
    // Allocate temporary host memory
    int *h_array = (int*)malloc(size * sizeof(int));
    if (h_array == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return;
    }

    // Copy data from device to host
    cudaError_t err = cudaMemcpy(h_array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(h_array);
        return;
    }

    // Print the array
    for (int i = 0; i < size; ++i) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    // Free temporary host memory
    free(h_array);
}

// Simple kernel to initialize device array with values 0..size-1
__global__ void init_array_kernel(int *d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = idx;
    }
}

int main(void) {
    const int size = 10;
    int *d_array = NULL;

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_array, size * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Initialize device array
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    init_array_kernel<<<blocks, threadsPerBlock>>>(d_array, size);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_array);
        return 1;
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Print device array
    print_device_array(d_array, size);

    // Clean up
    cudaFree(d_array);
    return 0;
}
