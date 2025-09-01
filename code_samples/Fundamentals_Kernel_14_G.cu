/*
Launch a kernel with a block dimension of 0. What error does this cause?

The program's aim is to launch a CUDA kernel with an invalid block dimension of zero to observe the runtime error reported by the CUDA runtime. 
In CUDA, block dimensions must be positive integers; zero is illegal. When a kernel launch is attempted with blockDim.x == 0 (or any zero dimension), the CUDA driver detects the invalid configuration and returns an error. 
The specific error is cudaErrorInvalidConfiguration (value 7), which is reported via cudaGetLastError() or cudaDeviceSynchronize(). 
This error is described as "invalid configuration argument". 
Thus, running the program should print something like:
   Kernel launch error: invalid configuration argument
The program will:
1. Define an empty kernel.
2. Set blockDim = (0,1,1) and gridDim = (1,1,1).
3. Launch the kernel.
4. Immediately query cudaGetLastError() to capture the error.
5. Print the error string.
6. Exit. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void emptyKernel() {
    // No operation
}

int main(void) {
    // Set block dimension to zero (invalid)
    dim3 blockDim(0, 1, 1);
    dim3 gridDim(1, 1, 1);

    // Launch kernel with invalid block dimension
    emptyKernel<<<gridDim, blockDim>>>();

    // Capture the error from the kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel launched successfully (unexpected).\n");
    }

    // Optional: synchronize to catch any additional errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Post-launch synchronization error: %s\n", cudaGetErrorString(err));
    }

    return 0;
}
