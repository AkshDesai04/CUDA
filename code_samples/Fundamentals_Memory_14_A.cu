/*
Aim of the program: Perform an out-of-bounds read instead of a write and analyze the `cuda-memcheck` report.

Thinking process:
- We want to trigger a memory error that is specifically an out-of-bounds read on the GPU.
- A simple way is to allocate an array of known size on device memory, then launch a kernel that attempts to read from an index beyond the allocated bounds.
- The kernel will read from d_arr[idx] where idx is larger than the array size. The read result will be stored into a separate output variable for inspection.
- After kernel launch, we copy the output back to host and print it. The actual value read is undefined, but the important part is that cuda-memcheck will report an out-of-bounds read error.
- We will include basic error checking for CUDA API calls to help identify launch failures, but the main purpose is to demonstrate memcheck's detection of read violations.
- The program is minimal and self-contained: it includes allocation, initialization, kernel launch, data transfer back to host, and printing.
- To run this with cuda-memcheck, one would compile with nvcc and then execute using `cuda-memcheck ./a.out`. The memcheck output should show an error such as "ERROR: out of bounds read of size X from device pointer 0x...".
- This example demonstrates that read violations are caught just as effectively as write violations, and that cuda-memcheck can help identify and locate the offending access.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE 10

__global__ void outOfBoundsReadKernel(const int *d_arr, int *d_out, int idx)
{
    // Attempt to read from an index that is intentionally out of bounds
    int value = d_arr[idx];      // idx == ARRAY_SIZE (out of bounds)
    d_out[0] = value;            // Store the read value for later inspection
}

int main(void)
{
    int hostArray[ARRAY_SIZE];
    int hostOut = 0;
    int *d_arr = NULL;
    int *d_out = NULL;

    // Initialize host array
    for (int i = 0; i < ARRAY_SIZE; ++i)
        hostArray[i] = i + 1;

    // Allocate device memory
    cudaError_t err = cudaMalloc((void **)&d_arr, ARRAY_SIZE * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_arr failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc((void **)&d_out, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_out failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return -1;
    }

    // Copy data from host to device
    err = cudaMemcpy(d_arr, hostArray, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy hostToDevice failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        cudaFree(d_out);
        return -1;
    }

    // Launch kernel with out-of-bounds index
    int outOfBoundsIdx = ARRAY_SIZE;  // This index is one past the last valid element
    outOfBoundsReadKernel<<<1, 1>>>(d_arr, d_out, outOfBoundsIdx);

    // Synchronize to catch any errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch or execution failed: %s\n", cudaGetErrorString(err));
    }

    // Copy result back to host
    err = cudaMemcpy(&hostOut, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy deviceToHost failed: %s\n", cudaGetErrorString(err));
    }

    // Print the (undefined) value read
    printf("Read value from out-of-bounds index: %d\n", hostOut);

    // Clean up
    cudaFree(d_arr);
    cudaFree(d_out);

    return 0;
}
