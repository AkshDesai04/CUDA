/*
Create another asynchronous error: an unspecified launch failure. E.g., a kernel that dereferences a null pointer.

We want to write a CUDA program that triggers an asynchronous error by dereferencing a null pointer inside a kernel. Since the kernel execution is asynchronous relative to the host, the error will not be reported until we query it with cudaGetLastError() or perform a cudaDeviceSynchronize(). By passing a null pointer to the kernel, we cause an illegal memory access in device code, leading to a launch failure. The program will launch the kernel, then immediately check for errors using cudaGetLastError, and also after synchronization. It will print out the CUDA error string for each check, demonstrating how asynchronous errors are handled in CUDA.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void errorKernel(int *ptr) {
    // Dereference null pointer
    *ptr = 42;
}

int main() {
    // Launch kernel with null pointer argument
    errorKernel<<<1, 1>>>(nullptr);
    
    // Check for launch errors immediately
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cudaGetLastError after launch: %s\n", cudaGetErrorString(err));
    } else {
        printf("cudaGetLastError after launch: No error (unexpected)\n");
    }
    
    // Synchronize to force error to propagate
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
    } else {
        printf("cudaDeviceSynchronize: No error (unexpected)\n");
    }
    
    return 0;
}
