/*
Perform an out-of-bounds read instead of a write and analyze the `cuda-memcheck` report.

Thinking:
1. The goal is to create a minimal CUDA C program that deliberately performs an out-of-bounds read
   on a device array. This allows us to observe how cuda-memcheck detects and reports such memory
   violations.
2. We will allocate a small integer array (e.g., size 10) on the device. A kernel will launch a
   single thread that attempts to read from index `size` (which is 10), which is one past the last
   valid index (0‑9). This is an out-of-bounds read.
3. The kernel will write the read value into another device location (a single-element array) so
   that the host can retrieve it and print it. Although the read is invalid, the write itself will
   be legal; we are only demonstrating the read violation.
4. Error checking for CUDA API calls and kernel launch will be performed.
5. After running this program under `cuda-memcheck`, we should see a report indicating an
   "Invalid read of size X" (typically 4 bytes for int) at the offending memory address.
6. The program prints the read value (though it's undefined) to illustrate that the kernel
   executed but the read was out-of-bounds.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that performs an out-of-bounds read
__global__ void outOfBoundsReadKernel(const int *d_arr, int size, int *d_out)
{
    // Only one thread will perform the read
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // Intentionally read one element past the end of the array
        int val = d_arr[size];   // Out-of-bounds read
        d_out[0] = val;          // Write the (undefined) value somewhere safe
    }
}

int main(void)
{
    const int N = 10;
    int h_arr[N];
    int h_out = 0;

    // Initialize host array
    for (int i = 0; i < N; ++i)
        h_arr[i] = i;

    // Device pointers
    int *d_arr = NULL;
    int *d_out = NULL;

    // Allocate device memory
    cudaError_t err = cudaMalloc((void **)&d_arr, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_arr failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    err = cudaMalloc((void **)&d_out, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_out failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return EXIT_FAILURE;
    }

    // Copy host array to device
    err = cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to device failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        cudaFree(d_out);
        return EXIT_FAILURE;
    }

    // Launch kernel: one block, one thread
    outOfBoundsReadKernel<<<1, 1>>>(d_arr, N, d_out);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        cudaFree(d_out);
        return EXIT_FAILURE;
    }

    // Wait for device to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        cudaFree(d_out);
        return EXIT_FAILURE;
    }

    // Copy the result back to host
    err = cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to host failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        cudaFree(d_out);
        return EXIT_FAILURE;
    }

    printf("Value read from out-of-bounds index: %d\n", h_out);

    // Clean up
    cudaFree(d_arr);
    cudaFree(d_out);

    return EXIT_SUCCESS;
}
