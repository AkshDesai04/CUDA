/*
Write a program that allocates a large amount of memory on the device (e.g., 512MB) and then immediately frees it using cudaFree. Use robust error checking for all API calls to ensure the allocation and deallocation were successful.

Thinking process:
- The goal is to demonstrate how to allocate and free a large chunk of GPU memory while performing error checks.
- We'll use `cudaMalloc` to allocate 512 megabytes (512 * 1024 * 1024 bytes).
- Immediately after allocation, we'll call `cudaFree` to deallocate the memory.
- To ensure robust error checking, we will wrap CUDA API calls in a macro `CHECK_CUDA` that checks the return status and prints a descriptive error message if the call fails, then exits the program.
- We'll include necessary headers (`<cuda_runtime.h>` and `<stdio.h>`).
- The main function will perform the allocation, check, deallocation, and check again, then print a success message.
- This code will be compiled as a CUDA C source file (.cu) and can be run on any system with CUDA installed.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                    \
        }                                                           \
    } while (0)

int main(void)
{
    const size_t size = 512 * 1024 * 1024; // 512 MB
    void* d_ptr = NULL;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_ptr, size));

    // Free the allocated memory
    CHECK_CUDA(cudaFree(d_ptr));

    printf("Successfully allocated and freed %zu bytes of device memory.\n", size);
    return EXIT_SUCCESS;
}
