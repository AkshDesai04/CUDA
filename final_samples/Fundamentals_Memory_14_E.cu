```cu
/*
Use `cudaDeviceSynchronize` and proper error checking. An out-of-bounds access often results in `cudaErrorIllegalAddress`. Catch and report this error.

Thought process:
- The aim is to demonstrate error detection after a kernel launch, especially for illegal memory accesses.
- We'll allocate a small array on the device and intentionally launch more threads than the array size.
- Each thread writes to an element indexed by its global thread ID. The last thread will write outside the bounds, causing an illegal address error.
- We use cudaGetLastError() to catch launch failures and cudaDeviceSynchronize() to detect runtime errors that occur during kernel execution.
- After synchronization, we check for cudaErrorIllegalAddress specifically and print a descriptive message.
- The program cleans up device memory before exiting.
*/

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void outOfBoundsKernel(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Intentionally write to out-of-bounds when idx == N
    if (idx < N + 1) {   // N+1 threads; last one will be out-of-bounds
        data[idx] = idx; // This will trigger cudaErrorIllegalAddress for idx == N
    }
}

int main(void) {
    const int N = 10;              // Size of the array
    int *d_data = NULL;
    cudaError_t err = cudaMalloc((void**)&d_data, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Launch N+1 threads intentionally to cause OOB access
    const int threadsPerBlock = 4;
    const int totalThreads = N + 1;
    const int blocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    outOfBoundsKernel<<<blocks, threadsPerBlock>>>(d_data, N);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return EXIT_FAILURE;
    }

    // Synchronize to catch runtime errors (e.g., illegal memory access)
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        if (err == cudaErrorIllegalAddress) {
            fprintf(stderr, "Runtime error: illegal memory access detected (cudaErrorIllegalAddress).\n");
        } else {
            fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        }
        cudaFree(d_data);
        return EXIT_FAILURE;
    }

    printf("Kernel executed successfully without illegal memory access.\n");

    cudaFree(d_data);
    return EXIT_SUCCESS;
}
```