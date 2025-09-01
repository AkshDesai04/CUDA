/*
Aim of the program: Write a function that takes an integer `N`, allocates an array of `N` floats on the device, and initializes all its elements to zero using `cudaMemset`. The function should return the device pointer.

Thinking:
1. We need to allocate device memory of size N * sizeof(float). Use `cudaMalloc`.
2. After allocation, we must set all bytes to zero. Since float elements are 4 bytes, `cudaMemset` will set each byte to zero, which effectively zeroes all floats.
3. The function should return the pointer to the allocated device memory. If allocation fails, return `nullptr` (or a null pointer) and optionally print an error.
4. Provide error checking after both `cudaMalloc` and `cudaMemset` calls.
5. Include a small `main` that demonstrates usage: allocate an array of some size, optionally copy it back to host to verify, then free the device memory.
6. Use standard CUDA runtime API (`cuda_runtime.h`) and standard IO for error messages.

The code below implements these steps in a single .cu file that can be compiled with `nvcc`.
*/

#include <cuda_runtime.h>
#include <stdio.h>

/**
 * @brief Allocates an array of N floats on the device and initializes it to zero.
 *
 * @param N Number of float elements to allocate.
 * @return Pointer to the allocated device array, or nullptr if allocation failed.
 */
float* allocateAndZeroFloatArray(int N)
{
    if (N <= 0) {
        fprintf(stderr, "Error: N must be positive.\n");
        return nullptr;
    }

    float* d_ptr = nullptr;
    size_t size_in_bytes = static_cast<size_t>(N) * sizeof(float);

    // Allocate device memory
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_ptr), size_in_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }

    // Initialize allocated memory to zero
    err = cudaMemset(d_ptr, 0, size_in_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_ptr);
        return nullptr;
    }

    return d_ptr;
}

int main()
{
    const int N = 10;
    float* d_array = allocateAndZeroFloatArray(N);
    if (d_array == nullptr) {
        fprintf(stderr, "Failed to allocate and zero device array.\n");
        return EXIT_FAILURE;
    }

    // Optional: copy back to host to verify
    float h_array[N];
    cudaMemcpy(h_array, d_array, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Device array after zeroing:\n");
    for (int i = 0; i < N; ++i) {
        printf("%f ", h_array[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_array);
    return EXIT_SUCCESS;
}