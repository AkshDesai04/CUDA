/*
Aim of the program:
Call `cudaDeviceSynchronize()` before `cudaDeviceReset()` and explain why this can be good practice.

Thinking behind this program:
1. The program launches a simple CUDA kernel that fills an array with values.
2. After kernel launch, it calls `cudaDeviceSynchronize()` to ensure that all pending GPU operations have finished before proceeding.
3. Only after synchronization does it call `cudaDeviceReset()` to cleanly reset the device.
4. Calling `cudaDeviceSynchronize()` before resetting is good practice because:
   - It guarantees that any asynchronous work (kernel launches, memory copies, etc.) has completed.
   - It prevents the device from being reset while operations are still in flight, which could lead to undefined behavior, crashes, or data corruption.
   - It ensures that any error status from the kernel execution is captured and can be reported before resetting the device.
   - It provides a clean separation between host and device states, making debugging and profiling easier.
5. The program also demonstrates basic error checking to illustrate how to handle CUDA errors in a robust way.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that writes the index into the output array
__global__ void simpleKernel(int *d_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_out[idx] = idx;
    }
}

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

int main(void) {
    const int N = 256;
    const size_t size = N * sizeof(int);
    int *h_out = (int *)malloc(size);
    int *d_out = NULL;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void **)&d_out, size));

    // Launch kernel
    int threadsPerBlock = 64;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    simpleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, N);

    // Synchronize to ensure all GPU work has finished before resetting device
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // Print first 10 results for verification
    printf("First 10 results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    // Clean up device memory
    CHECK_CUDA(cudaFree(d_out));

    // Reset device after synchronization
    CHECK_CUDA(cudaDeviceReset());

    // Free host memory
    free(h_out);

    printf("Program completed successfully.\n");
    return 0;
}
