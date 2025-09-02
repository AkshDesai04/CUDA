/*
Aim: Write a program that allocates memory on the GPU, then calls cudaDeviceReset(), and then attempts to use or free that memory. Observe the error that occurs.

Thinking:
- First we allocate some memory on the GPU using cudaMalloc.
- We then reset the device with cudaDeviceReset(), which destroys the current CUDA context and frees all device resources.
- After the reset, any device pointer we obtained before the reset becomes invalid because the context that allocated it no longer exists.
- Attempting to use that pointer (e.g., via cudaMemcpy) or freeing it with cudaFree should fail and return an error code.
- We capture and print the error string returned by the CUDA runtime after each call to illustrate the failure.
- The expected error after cudaMemcpy and cudaFree is something along the lines of "invalid device pointer" or "invalid context".
- The code is written in plain CUDA C and can be compiled with `nvcc`. It outputs the error messages to standard output.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int *d_ptr = NULL;
    size_t size = 100 * sizeof(int);

    // Allocate memory on the GPU
    cudaError_t err = cudaMalloc((void**)&d_ptr, size);
    printf("cudaMalloc: %s\n", cudaGetErrorString(err));

    // Reset the device (destroys context and frees resources)
    err = cudaDeviceReset();
    printf("cudaDeviceReset: %s\n", cudaGetErrorString(err));

    // Attempt to use the memory after reset (should fail)
    int h_val = 0;
    err = cudaMemcpy(&h_val, d_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    printf("cudaMemcpy after reset: %s\n", cudaGetErrorString(err));

    // Attempt to free the memory after reset (should also fail)
    err = cudaFree(d_ptr);
    printf("cudaFree after reset: %s\n", cudaGetErrorString(err));

    return 0;
}
