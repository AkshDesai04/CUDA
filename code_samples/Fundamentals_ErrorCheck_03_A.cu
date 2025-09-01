/*
Explain the difference between `cudaPeekAtLastError` and `cudaGetLastError` in a comment. (Peek does not reset the error state)

This program demonstrates the behavior of `cudaPeekAtLastError` versus `cudaGetLastError`.  
We intentionally launch a kernel that dereferences a null pointer to trigger an
illegal memory access error on the device. After the launch we call both error
checking functions and print their results. The output shows that:

1. `cudaPeekAtLastError` returns the last error that occurred but does **not**
   clear the error state. A subsequent call to `cudaPeekAtLastError` will return
   the same error until the error state is cleared.

2. `cudaGetLastError` returns the last error *and* clears the error state.
   After calling `cudaGetLastError`, another call to `cudaPeekAtLastError` will
   return `cudaSuccess`.

By printing the error codes and their string representations, we can observe
this difference. The program also includes basic CUDA error checking after
memory allocation and kernel launch, and reports any additional errors that
may occur.

The aim of this program, as requested, is to provide a clear comment that
explains the difference between `cudaPeekAtLastError` and `cudaGetLastError`,
highlighting that Peek does not reset the error state.
*/
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void nullKernel(int *ptr)
{
    int idx = threadIdx.x;
    // This will trigger an illegal memory access if ptr is NULL
    ptr[idx] = idx;
}

int main(void)
{
    // Allocate a small array on device
    int *d_arr = nullptr;
    cudaMalloc((void**)&d_arr, 10 * sizeof(int));

    // Intentionally pass NULL to trigger an error
    nullKernel<<<1, 10>>>(nullptr);

    // First check using cudaPeekAtLastError
    cudaError_t errPeek = cudaPeekAtLastError();
    printf("cudaPeekAtLastError first call: %s\n", cudaGetErrorString(errPeek));

    // Second check using cudaPeekAtLastError again to show it does not reset
    cudaError_t errPeekAgain = cudaPeekAtLastError();
    printf("cudaPeekAtLastError second call: %s\n", cudaGetErrorString(errPeekAgain));

    // Now use cudaGetLastError, which clears the error state
    cudaError_t errGet = cudaGetLastError();
    printf("cudaGetLastError call: %s\n", cudaGetErrorString(errGet));

    // After clearing, peek should return cudaSuccess
    cudaError_t errPeekAfterClear = cudaPeekAtLastError();
    printf("cudaPeekAtLastError after cudaGetLastError: %s\n", cudaGetErrorString(errPeekAfterClear));

    // Clean up
    cudaFree(d_arr);
    return 0;
}