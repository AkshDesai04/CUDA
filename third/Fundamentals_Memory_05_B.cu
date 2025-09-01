```cuda
/* 
Aim of the program: 
Try to call `cudaFree` on the same pointer twice. What happens? Use error checking to see the result of the second call. 

Thinking: 
- We want a minimal CUDA C program that demonstrates what occurs when freeing a device pointer twice.
- Allocate a device pointer using `cudaMalloc`. 
- Free it the first time. `cudaFree` should return `cudaSuccess`.
- Attempt to free the same pointer again. According to CUDA documentation, freeing an already freed or invalid pointer should return `cudaErrorInvalidDevicePointer` (or possibly `cudaErrorInvalidValue` if the pointer is `NULL`).
- We will capture the return value of each `cudaFree` call and print an error message if the status is not `cudaSuccess`. 
- Additionally, after each call we can use `cudaGetLastError()` to see if any error was recorded, but the return value is sufficient for this demonstration.
- Finally, we exit. No kernel launch is needed for this simple test.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int *d_ptr = NULL;
    size_t size = 10 * sizeof(int);

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("cudaMalloc succeeded, pointer = %p\n", (void*)d_ptr);

    // First cudaFree
    err = cudaFree(d_ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "First cudaFree failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("First cudaFree succeeded.\n");
    }

    // Second cudaFree on the same pointer
    err = cudaFree(d_ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "Second cudaFree failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("Second cudaFree succeeded (unexpected).\n");
    }

    // Check for any lingering errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError after frees: %s\n", cudaGetErrorString(err));
    } else {
        printf("No lingering CUDA errors.\n");
    }

    return 0;
}
```