/*
Aim: Add comments explaining that `cudaDeviceReset()` destroys all allocations and contexts on the current device.

Thinking:
- The goal is to illustrate how `cudaDeviceReset()` works and to document its effect in comments.
- We allocate device memory, launch a trivial kernel, copy results back, and then free the memory manually.
- Before calling `cudaDeviceReset()`, we include a clear comment that explains it destroys all allocations and contexts on the current device.
- After the reset, any use of the freed pointers would be invalid, so we comment out an example that would cause an error if uncommented.
- The program remains simple but demonstrates the key points required by the prompt.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void simpleKernel(int *d_arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        d_arr[idx] = idx;
}

int main()
{
    int n = 10;
    int *d_arr = NULL;

    // Allocate device memory
    cudaMalloc((void**)&d_arr, n * sizeof(int));

    // Launch kernel
    simpleKernel<<<(n + 255)/256, 256>>>(d_arr, n);
    cudaDeviceSynchronize();

    // Copy results back to host
    int h_arr[10];
    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Kernel result:\n");
    for (int i = 0; i < n; ++i)
        printf("%d ", h_arr[i]);
    printf("\n");

    // Clean up allocations manually
    cudaFree(d_arr);

    // Reset the device. This call destroys all allocations and contexts
    // on the current device. After this, any pointer to device memory
    // (like d_arr) becomes invalid, and you cannot launch kernels or
    // perform memory operations until you reinitialize the device
    // context by calling cudaSetDevice or similar.
    cudaDeviceReset();

    // If we attempt to use d_arr after reset, it will result in an error
    // because the allocation no longer exists. Uncommenting the following
    // lines would lead to a segmentation fault or CUDA error.
    /*
    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    */

    printf("Device reset complete.\n");
    return 0;
}
