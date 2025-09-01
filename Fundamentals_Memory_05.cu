/*
Write a program that allocates a large amount of memory on the device (e.g., 512MB) and then immediately frees it using `cudaFree`. Use robust error checking for all API calls to ensure the allocation and deallocation were successful.

Thinking:
- The goal is to demonstrate dynamic allocation and deallocation on the GPU, with thorough error handling.
- We'll use `cudaMalloc` to request 512 MiB of device memory.
- The size is computed as `512 * 1024 * 1024` bytes, ensuring alignment with the 1 MiB units.
- Error checking is performed on every CUDA API call via a helper macro `CUDA_CHECK`.
- After allocation, we immediately free the memory with `cudaFree` and again check for errors.
- The program prints a success message if both operations succeed.
- No kernel launches or device operations are needed; this is purely an allocation test.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main(void) {
    const size_t memSize = 512ull * 1024 * 1024; // 512 MB
    void* d_ptr = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_ptr, memSize));
    printf("Successfully allocated %zu bytes on the device.\n", memSize);

    // Immediately free the allocated memory
    CUDA_CHECK(cudaFree(d_ptr));
    printf("Successfully freed the allocated memory.\n");

    return EXIT_SUCCESS;
}
