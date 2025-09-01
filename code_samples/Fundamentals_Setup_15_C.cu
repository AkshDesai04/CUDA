/*
Add comments explaining that `cudaDeviceReset()` destroys all allocations and contexts on the current device.

Below is the entire thought process and plan for the program:

1. The goal is to illustrate the use of `cudaDeviceReset()` and to add comments that explain its behavior.  
2. We'll write a minimal CUDA program that:
   - Allocates a small array on the host.
   - Copies it to device memory.
   - Performs a trivial kernel that increments each element.
   - Copies the result back to the host and prints it.
3. After the kernel execution and data copy, we'll call `cudaDeviceReset()` to free all resources on the current device.  
4. We will add a comment right before the call to `cudaDeviceReset()` that states: "This call destroys all allocations and contexts on the current device, effectively cleaning up the GPU state."  
5. Finally, we will print a message indicating that the device has been reset successfully.

The program is self‑contained and can be compiled with `nvcc` and run on a system with CUDA installed.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that increments each element of an array by 1
__global__ void incrementKernel(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] += 1;
    }
}

int main(void) {
    const int N = 10;
    int h_arr[N];
    int *d_arr = NULL;

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;
    }

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_arr, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy data from host to device
    err = cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (H2D) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    incrementKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (D2H) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    // Print results
    printf("Result after kernel execution:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_arr);

    /* This call destroys all allocations and contexts on the current device, effectively cleaning up the GPU state. */
    err = cudaDeviceReset();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Device reset successfully.\n");
    return 0;
}
