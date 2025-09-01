/*
Aim: Add `cudaDeviceSynchronize()` after the launch and print 'Kernel synchronized!' after it.
Thinking:
The goal is to demonstrate how to properly synchronize a CUDA kernel launch with the host thread.
In a typical CUDA program, after launching a kernel we often need to ensure that the kernel has finished executing
before the host proceeds, especially when we want to check for errors or use the results. 
The CUDA runtime API provides cudaDeviceSynchronize() for this purpose, which blocks the host until all preceding 
device work is complete. After calling it, we can safely print a message indicating that synchronization is done.
To illustrate this, I will write a minimal CUDA program that:
1. Allocates a small array on the device.
2. Launches a simple kernel that writes a constant value to each element.
3. Calls cudaDeviceSynchronize() right after the kernel launch.
4. Prints "Kernel synchronized!".
5. Copies the data back to host and prints it to verify.
The code includes error checking after kernel launch and after the synchronization call.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that writes a constant value to each element of an array
__global__ void writeConst(int *d_arr, int val, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_arr[idx] = val;
    }
}

int main() {
    const int N = 16;
    const int val = 42;
    int h_arr[N];

    int *d_arr = NULL;
    size_t size = N * sizeof(int);

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_arr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Define block and grid dimensions
    int threadsPerBlock = 8;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    writeConst<<<blocksPerGrid, threadsPerBlock>>>(d_arr, val, N);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    // Synchronize the device to ensure kernel completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    // Print synchronization confirmation
    printf("Kernel synchronized!\n");

    // Copy result back to host
    err = cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    // Verify and print the result
    printf("Result array:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_arr);
    return 0;
}
