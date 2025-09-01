/*
Aim of the program: Launch a kernel with a grid dimension of 0. What error does this cause?

Thinking: 
- In CUDA, launching a kernel requires non-zero grid and block dimensions.
- Setting grid dimension to zero is invalid.
- The runtime should return an error after the launch, typically CUDA_ERROR_INVALID_VALUE (error 11).
- We'll create a dummy kernel that does nothing.
- In main, set grid to dim3(0) and block to dim3(1).
- Launch kernel, then call cudaGetLastError to capture error.
- Print the error message and code.
- Also call cudaDeviceSynchronize to ensure error handling.
- This program demonstrates that launching with grid=0 causes an invalid configuration argument error.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dummyKernel()
{
    // No operation
}

int main()
{
    dim3 grid(0); // Intentional invalid grid dimension
    dim3 block(1);

    // Launch kernel
    dummyKernel<<<grid, block>>>();

    // Retrieve error from the launch
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        printf("Kernel launch failed with error: %s (code %d)\n",
               cudaGetErrorString(err), err);
    } else {
        printf("Kernel launch succeeded (unexpected).\n");
    }

    // Synchronize to ensure any pending errors are reported
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("cudaDeviceSynchronize failed with error: %s (code %d)\n",
               cudaGetErrorString(err), err);
    } else {
        printf("cudaDeviceSynchronize succeeded.\n");
    }

    return 0;
}
