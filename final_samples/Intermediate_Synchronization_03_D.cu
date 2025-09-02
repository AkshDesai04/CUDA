```cuda
/* 
Aim: Create a race condition on an array: have all threads increment `output[threadIdx.x % 4]`. Some locations will have races, others won't.

Thinking: 
The program demonstrates a classic data race scenario on a small array of size 4. 
Each thread in the kernel calculates the target index as `threadIdx.x % 4` and performs an increment on that element without any synchronization (no atomic or mutex). 
Because multiple threads can target the same index simultaneously, the final value of that element becomes undefined. 
By choosing a block dimension that is not a multiple of 4 (e.g., 7), we can observe that indices 0, 1, and 2 receive increments from two threads each, while index 3 is only touched by one thread. 
Thus, indices 0–2 experience races and may show incorrect counts, whereas index 3 is race‑free. 
The host code initializes the array, launches the kernel, copies the results back, and prints them to illustrate the effect. 
All device memory operations include basic error checking for clarity.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that introduces a race condition by incrementing output[threadIdx.x % 4]
__global__ void raceKernel(int *output)
{
    // Each thread computes an index modulo 4 and increments that location
    int idx = threadIdx.x % 4;
    output[idx]++;    // No atomic, so concurrent updates race
}

int main()
{
    const int N = 4;              // Size of the output array
    const int threadsPerBlock = 7; // Non‑multiple of 4 to show race-free element

    int h_output[N] = {0};        // Host array initialized to zero

    // Allocate device memory
    int *d_output;
    cudaError_t err = cudaMalloc((void**)&d_output, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy initial zeroed array to device
    err = cudaMemcpy(d_output, h_output, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_output);
        return 1;
    }

    // Launch the kernel
    raceKernel<<<1, threadsPerBlock>>>(d_output);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_output);
        return 1;
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy results back to host
    err = cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_output);
        return 1;
    }

    // Free device memory
    cudaFree(d_output);

    // Print the results
    printf("Resulting array values after %d threads:\n", threadsPerBlock);
    for (int i = 0; i < N; ++i) {
        printf("output[%d] = %d\n", i, h_output[i]);
    }

    return 0;
}
```