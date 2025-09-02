```cpp
/*
 * Aim of the program: Write a program that allocates 10 small chunks of memory and stores their pointers in an array. Then, loop through the array to free them all.
 *
 * My thinking:
 * 1. The program will run on the host and use CUDA runtime API to allocate memory on the GPU.
 * 2. We'll declare an array of 10 pointers (e.g., `int* ptrs[10];`) to hold the device memory addresses.
 * 3. For each pointer, we call `cudaMalloc` to allocate a small chunk (e.g., 100 bytes).
 * 4. We'll check the return status of each call and print an error message if allocation fails.
 * 5. After allocations, we'll iterate over the array again, calling `cudaFree` for each pointer.
 * 6. We'll also check the status of `cudaFree` and report any errors.
 * 7. Finally, we call `cudaDeviceReset()` to clean up the device before exiting.
 *
 * The program uses the CUDA runtime API, so it should be compiled with `nvcc`.
 * No device kernels are launched; this is purely host code exercising memory allocation and deallocation.
 */

#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    const int num_chunks = 10;
    const size_t chunk_size = 100;  // bytes
    int* ptrs[num_chunks];

    // Allocate memory chunks
    for (int i = 0; i < num_chunks; ++i) {
        cudaError_t err = cudaMalloc((void**)&ptrs[i], chunk_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for chunk %d: %s\n", i, cudaGetErrorString(err));
            // Attempt to free any previously allocated chunks before exiting
            for (int j = 0; j < i; ++j) {
                cudaFree(ptrs[j]);
            }
            return 1;
        } else {
            printf("Allocated chunk %d at device pointer %p\n", i, (void*)ptrs[i]);
        }
    }

    // Free memory chunks
    for (int i = 0; i < num_chunks; ++i) {
        cudaError_t err = cudaFree(ptrs[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaFree failed for chunk %d: %s\n", i, cudaGetErrorString(err));
            // Continue freeing the rest even if one fails
        } else {
            printf("Freed chunk %d\n", i);
        }
    }

    // Reset the device to clean up resources
    cudaDeviceReset();
    return 0;
}
```