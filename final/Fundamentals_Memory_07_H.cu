```cuda
/*
Aim of the program: Explain how `cudaMemcpy` bridges the gap between these two separate memory spaces.

Thought Process:
1. The user wants an explanation of how cudaMemcpy works between host (CPU) and device (GPU) memory spaces.
2. To illustrate this, we write a simple CUDA C program that:
   - Allocates an array on the host.
   - Allocates a corresponding array on the device.
   - Uses cudaMemcpy to transfer data from host to device.
   - Launches a trivial kernel that modifies the data on the device.
   - Uses cudaMemcpy again to bring the data back to the host.
3. This demonstrates the two distinct memory spaces and how cudaMemcpy bridges them.
4. We also add basic error checking for cudaMemcpy and kernel launch.
5. The program outputs the modified data to show that the transfer was successful.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that increments each element by 1
__global__ void increment_kernel(int *d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_data[idx] += 1;
    }
}

int main(void) {
    const int N = 10;
    int h_data[N];
    int *d_data;

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;
    }

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_data, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy data from host to device
    err = cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (HostToDevice) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Launch kernel: one block of N threads
    increment_kernel<<<1, N>>>(d_data, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Copy result back from device to host
    err = cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (DeviceToHost) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Print the results
    printf("Modified data after kernel execution:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_data);
    return 0;
}
```