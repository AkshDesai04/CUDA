/*
Launch a kernel with a block dimension of 0. What error does this cause?

Thinking:
The purpose of this program is to illustrate the CUDA runtime error that occurs when a kernel is launched with an invalid block dimension of zero. In CUDA, block and grid dimensions must be positive integers; setting a block dimension to zero is illegal and should trigger the CUDA error code cudaErrorInvalidConfiguration (error code 11). 
To demonstrate this, we will:
1. Define a simple no-op kernel.
2. In the host code, launch this kernel with a grid dimension of 1 and a block dimension of 0.
3. Immediately query cudaGetLastError() to capture the error produced by the kernel launch.
4. Print the error string to confirm that it is indeed "invalid configuration argument".
5. Optionally, perform cudaDeviceSynchronize() to catch any further errors (though the launch error is already captured).
6. The program is self‑contained, includes necessary headers, and can be compiled with nvcc.
*/

#include <iostream>
#include <cuda_runtime.h>

__global__ void dummyKernel() {
    // No operation
}

int main() {
    // Attempt to launch kernel with block dimension of 0
    dummyKernel<<<1, 0>>>();

    // Capture error from kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "Kernel launched successfully." << std::endl;
    }

    // Optional: synchronize to catch any pending errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "Device synchronize error: " << cudaGetErrorString(err) << std::endl;
    }

    return 0;
}
