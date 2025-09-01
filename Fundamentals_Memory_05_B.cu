```cpp
/*
Aim: Try to call `cudaFree` on the same pointer twice. What happens? Use error checking to see the result of the second call.

Thinking:
The goal is to demonstrate that freeing the same device pointer twice results in an error. 
In CUDA, `cudaFree` returns a `cudaError_t`. After the first call, the memory block is deallocated and the pointer is no longer valid. 
Calling `cudaFree` again on the same address should produce an error such as `cudaErrorInvalidDevicePointer` or a similar error code. 
We will allocate a small integer on the device, free it once, and then attempt to free it again. 
After each call we will check the return status and print both the numeric error code and the human‑readable error string provided by `cudaGetErrorString`. 
This will illustrate the error handling mechanism for repeated frees. 
The program is written in C/C++ using CUDA runtime API, compiled with `nvcc` as a `.cu` file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for checking CUDA API call errors
#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error in file '%s' at line %d: %s (%d)\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err), err); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

int main(void)
{
    int *d_ptr = NULL;
    size_t size = sizeof(int);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_ptr, size));
    printf("Allocated %zu bytes on device at %p\n", size, (void*)d_ptr);

    // First cudaFree call
    cudaError_t err1 = cudaFree(d_ptr);
    if (err1 == cudaSuccess) {
        printf("First cudaFree succeeded.\n");
    } else {
        printf("First cudaFree failed with error: %s (%d)\n",
               cudaGetErrorString(err1), err1);
    }

    // Second cudaFree call on the same pointer
    cudaError_t err2 = cudaFree(d_ptr);
    if (err2 == cudaSuccess) {
        printf("Second cudaFree succeeded (unexpected).\n");
    } else {
        printf("Second cudaFree failed with error: %s (%d)\n",
               cudaGetErrorString(err2), err2);
    }

    // Reset the device (optional but clean)
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
```