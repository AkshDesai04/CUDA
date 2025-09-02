/*
In the version with error checking, use your `CHECK` macro to catch and report the `cudaErrorIllegalAddress`.

Thinking:
- We need a small CUDA program that deliberately triggers an illegal address error.
- A common way is to write or read beyond the bounds of a device array.
- Allocate an array of size N, launch a kernel that writes to d_arr[N], which is out-of-bounds.
- The kernel launch itself will succeed, but the runtime error will surface when the kernel executes.
- To capture both launch and execution errors, call CHECK(cudaGetLastError()) after the launch, and CHECK(cudaDeviceSynchronize()) afterwards.
- The CHECK macro will wrap a CUDA API call, check the returned cudaError_t, and if not cudaSuccess, print an error message with file, line, and the CUDA error string, then exit.
- The program will therefore report the illegal address error via the CHECK macro.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call)                                           \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            fprintf(stderr,                                   \
                    "CUDA error at %s:%d: %s\n",              \
                    __FILE__, __LINE__,                       \
                    cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

// Kernel that intentionally writes beyond array bounds
__global__ void illegalKernel(int *d_arr, int N)
{
    // Write to the element at index N, which is out of bounds
    d_arr[N] = 123;
}

int main(void)
{
    const int N = 10;
    int *d_arr = NULL;

    // Allocate device memory
    CHECK(cudaMalloc((void**)&d_arr, N * sizeof(int)));

    // Launch kernel with one thread to trigger the illegal write
    illegalKernel<<<1, 1>>>(d_arr, N);

    // Check for launch errors
    CHECK(cudaGetLastError());

    // Synchronize to catch execution errors such as illegal address
    CHECK(cudaDeviceSynchronize());

    // Free device memory
    CHECK(cudaFree(d_arr));

    printf("Program completed without encountering an illegal address.\n");
    return 0;
}
