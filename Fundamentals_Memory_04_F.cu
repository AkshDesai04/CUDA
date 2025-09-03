/*
Write a function that takes a device pointer and a size and calls `cudaMemset` to zero it out.

My thinking:

1. The goal is to provide a simple CUDA helper that wraps `cudaMemset`.  
2. The function will take a `void* d_ptr` and a `size_t size`.  
3. We'll call `cudaMemset(d_ptr, 0, size)` and check for errors.  
4. For demonstration purposes, the .cu file will contain a small `main` that allocates device memory, calls the helper, copies the result back to the host, and prints a few bytes to confirm they are zero.  
5. Error handling: if `cudaMemset` fails, we print the error string and exit.  
6. We include necessary headers: `<cuda_runtime.h>`, `<stdio.h>`.  
7. The code will compile with `nvcc`.  
8. Since the prompt explicitly wants only the .cu file, nothing else is output.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Helper function that zeros out a device buffer of given size
void cudaZeroBuffer(void* d_ptr, size_t size) {
    // Call cudaMemset to set all bytes to zero
    cudaError_t err = cudaMemset(d_ptr, 0, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Simple test to demonstrate usage
int main(void) {
    const size_t N = 64;               // Number of floats
    const size_t bytes = N * sizeof(float);

    // Allocate device memory
    float* d_arr = NULL;
    cudaError_t err = cudaMalloc((void**)&d_arr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Allocate host memory and fill with non-zero values
    float* h_arr = (float*)malloc(bytes);
    for (size_t i = 0; i < N; ++i) {
        h_arr[i] = (float)(i + 1);
    }

    // Copy data to device (not strictly needed for zeroing, but useful to show prior state)
    err = cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Zero out the device buffer using the helper
    cudaZeroBuffer(d_arr, bytes);

    // Copy back to host to verify
    err = cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Print first 8 elements to verify they are zero
    printf("First 8 elements after cudaMemset:\n");
    for (size_t i = 0; i < 8; ++i) {
        printf("%f ", h_arr[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_arr);
    free(h_arr);

    return EXIT_SUCCESS;
}
