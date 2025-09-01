/*
Pass an integer literal (e.g., 5) as an argument to the empty kernel.

Thinking: The user wants a simple CUDA program that demonstrates how to pass an integer literal to a kernel. 
We'll define a kernel that takes an int parameter but does nothing. 
On the host side we will launch this kernel with a literal value, e.g., 5. 
We'll include <stdio.h> for printing from host, and use cudaDeviceSynchronize() to ensure kernel completion.
We will also check for errors after kernel launch. 
This will compile as a .cu file. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void emptyKernel(int val) {
    // Kernel does nothing but can optionally use val
}

int main() {
    // Launch kernel with integer literal 5
    emptyKernel<<<1,1>>>(5);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA sync failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("Kernel launched with integer literal 5.\n");
    return 0;
}