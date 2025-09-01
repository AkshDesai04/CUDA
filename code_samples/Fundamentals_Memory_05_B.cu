/*
Aim: Try to call `cudaFree` on the same pointer twice. What happens? Use error checking to see the result of the second call.

Thought Process:
1. The goal is to demonstrate that freeing a CUDA device pointer more than once leads to an error.
2. We'll allocate a small array on the GPU, free it, and then attempt to free it again.
3. CUDA provides error codes returned by API functions. The second call to `cudaFree` should not succeed; we expect an error such as `cudaErrorInvalidDevicePointer`.
4. We'll write an error‑checking helper that prints the error string and the line number if a CUDA call fails.
5. The program will:
   - Allocate memory (`cudaMalloc`).
   - Free it once (`cudaFree`).
   - Attempt a second free and capture the returned error code.
6. Finally, we print the outcome of each call, which should clearly show the first free succeeded and the second returned an error.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main(void) {
    int *d_ptr = nullptr;
    size_t size = 10 * sizeof(int);

    // Allocate memory on the device
    printf("Allocating %zu bytes on the device...\n", size);
    CHECK_CUDA(cudaMalloc(&d_ptr, size));
    printf("Allocation successful. Pointer = %p\n", (void*)d_ptr);

    // First cudaFree call
    printf("Calling cudaFree for the first time...\n");
    cudaError_t err1 = cudaFree(d_ptr);
    if (err1 == cudaSuccess) {
        printf("First cudaFree succeeded.\n");
    } else {
        printf("First cudaFree failed with error: %s\n", cudaGetErrorString(err1));
    }

    // Second cudaFree call on the same pointer
    printf("Calling cudaFree for the second time...\n");
    cudaError_t err2 = cudaFree(d_ptr);
    if (err2 == cudaSuccess) {
        printf("Second cudaFree succeeded (unexpected).\n");
    } else {
        printf("Second cudaFree failed with error: %s\n", cudaGetErrorString(err2));
    }

    return 0;
}
