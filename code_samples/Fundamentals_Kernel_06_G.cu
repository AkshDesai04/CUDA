/*
Launch the kernel with 10 threads, but have all 10 threads try to write to `data[0]`. What value do you expect to see when you copy it back? (This is a race condition).

Thinking process:
- The goal is to demonstrate a classic data race where multiple CUDA threads concurrently write to the same memory location.
- In CUDA, threads in a block execute concurrently; without any synchronization or atomic operation, writes to the same address from multiple threads are not ordered.
- We will create a kernel where each thread writes its own threadIdx.x value to data[0].
- The kernel will be launched with 1 block of 10 threads. All threads attempt to write to data[0] simultaneously.
- After kernel execution, we copy the result back to the host and print it.
- Because of the race condition, the final value of data[0] will be nondeterministic; it could be any integer from 0 to 9 depending on the order of write execution.
- The program will compile with nvcc and run on any GPU that supports CUDA.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel where every thread writes its threadIdx.x to data[0]
__global__ void raceKernel(int *data) {
    // All 10 threads write to the same memory location
    data[0] = threadIdx.x;
}

int main() {
    int *d_data;
    int h_data = 0;

    // Allocate memory on device
    cudaError_t err = cudaMalloc((void**)&d_data, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Launch kernel with 1 block of 10 threads
    raceKernel<<<1, 10>>>(d_data);

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return -1;
    }

    // Copy result back to host
    err = cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return -1;
    }

    // Print the value read back
    printf("Value in data[0] after kernel launch: %d\n", h_data);

    // Clean up
    cudaFree(d_data);
    return 0;
}
