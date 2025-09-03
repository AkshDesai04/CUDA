/*
Aim of the program: Modify the kernel to write the value of its `threadIdx.x` to `data[0]`. Launch with one thread and verify.

Thinking:
1. We need a simple CUDA kernel that takes a pointer to an integer array on device memory.
2. Inside the kernel, we will write the value of `threadIdx.x` to the first element of the array: `data[0] = threadIdx.x;`.
3. In `main()`, allocate a single integer on the device (`int* d_data`).
4. Launch the kernel with one block of one thread: `<<<1, 1>>>`.
5. After kernel execution, copy the result back to host memory and print it to verify that it matches 0 (since threadIdx.x is 0 for the single thread).
6. Include basic error checking to ensure correct execution.
*/

#include <cuda_runtime.h>
#include <stdio.h>

// Kernel that writes threadIdx.x to data[0]
__global__ void writeThreadIdx(int *data) {
    data[0] = threadIdx.x;
}

int main() {
    int *d_data;
    int h_data;

    // Allocate device memory for one integer
    cudaError_t err = cudaMalloc((void**)&d_data, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernel with one block and one thread
    writeThreadIdx<<<1, 1>>>(d_data);

    // Wait for kernel to finish and check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Verify and print the result
    printf("ThreadIdx.x written to data[0] on device: %d\n", h_data);

    // Clean up
    cudaFree(d_data);
    return 0;
}
