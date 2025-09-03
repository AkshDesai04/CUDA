/*
Aim: Modify the kernel so that only threads with an even index print their ID.
Thinking:
The task is to adjust an existing CUDA kernel so that it filters out odd‑indexed threads and only prints the IDs of even ones. 
Each thread can compute its global linear index as `blockIdx.x * blockDim.x + threadIdx.x`. By checking `idx % 2 == 0` we can determine whether the index is even. 
Inside the kernel we will guard the `printf` call with this condition. 
In the host code we set up a simple launch configuration (e.g., 4 blocks of 256 threads each) and call the kernel. 
After the launch, we call `cudaDeviceSynchronize()` to ensure all prints are flushed before the program exits. 
Error checking can be added for robustness but is optional for this minimal example. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printEvenThread() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx % 2 == 0) {
        printf("Thread ID: %d\n", idx);
    }
}

int main() {
    int threadsPerBlock = 256;
    int blocks = 4;

    printEvenThread<<<blocks, threadsPerBlock>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
