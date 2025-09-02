```cpp
/*
Aim of the program: Launch the kernel with 256 threads. The expected result is 256. What result do you actually get? (Something much smaller).

Thinking:
- We want to illustrate why a naive increment of a global counter from many threads does not yield the expected count.
- The kernel will increment a global integer variable without using atomic operations, leading to race conditions.
- We launch one block with 256 threads, expecting that each thread will increment the counter once, resulting in 256.
- In practice, due to non‑atomic writes, many increments will be lost, and the final value will be much smaller.
- The code will:
  1. Allocate a single integer on the device and set it to 0.
  2. Launch a kernel that each thread performs `counter++`.
  3. Copy the result back to the host and print it.
- We also check for kernel launch errors with `cudaGetLastError`.
- The program is intentionally simple to highlight the issue.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void incrementKernel(int *counter)
{
    // Each thread attempts to increment the same counter
    // This is NOT atomic, so many increments will be lost
    (*counter)++;
}

int main(void)
{
    int *d_counter;
    int h_counter = 0;
    cudaError_t err;

    // Allocate device memory for the counter
    err = cudaMalloc((void **)&d_counter, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Initialize counter to 0 on the device
    err = cudaMemset(d_counter, 0, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_counter);
        return 1;
    }

    // Launch kernel with 256 threads in a single block
    const int threadsPerBlock = 256;
    incrementKernel<<<1, threadsPerBlock>>>(d_counter);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_counter);
        return 1;
    }

    // Wait for the device to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_counter);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_counter);
        return 1;
    }

    // Print the observed result
    printf("Expected result: 256\n");
    printf("Actual result:   %d\n", h_counter);

    // Clean up
    cudaFree(d_counter);
    return 0;
}
```