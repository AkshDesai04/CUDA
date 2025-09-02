/*
Write a helper function `void host_to_device_copy(int* h_data, int* d_data, int size)` that encapsulates the `cudaMemcpy` call.

Thinking process:
1. Include the necessary CUDA runtime header (`cuda_runtime.h`) and standard I/O for error reporting.
2. Define the helper function with the exact signature.
3. Inside the function, perform the copy using `cudaMemcpy` with `cudaMemcpyHostToDevice` direction.
4. Capture the return status and print an error message if the copy fails.
5. Provide a small `main` function to illustrate usage and ensure the file compiles as a complete CUDA source file.
*/

#include <cuda_runtime.h>
#include <stdio.h>

void host_to_device_copy(int* h_data, int* d_data, int size) {
    cudaError_t err = cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying from host to device: %s\n", cudaGetErrorString(err));
    }
}

int main() {
    // Example usage
    int size = 10;
    int h_data[10];
    for (int i = 0; i < size; i++) h_data[i] = i;
    int *d_data;
    cudaMalloc(&d_data, size * sizeof(int));

    host_to_device_copy(h_data, d_data, size);

    // Clean up
    cudaFree(d_data);
    return 0;
}
