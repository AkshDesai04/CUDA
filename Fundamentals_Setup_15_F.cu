/*
Call `cudaDeviceSynchronize()` before `cudaDeviceReset()` and explain why this can be good practice.
This program demonstrates why synchronizing the device before resetting it is advisable.
The comment below explains the reasoning:

When a CUDA program launches kernels or performs memory copies, these operations are
asynchronous with respect to the host.  The CUDA runtime queues them on the GPU
and immediately returns control to the host thread.  If we call `cudaDeviceReset()`
directly after launching a kernel, the runtime will destroy the context and free
all resources, possibly while the kernel is still executing or pending.  This can
lead to undefined behavior, lost errors, and hard‑to‑debug crashes.  By calling
`cudaDeviceSynchronize()` first, we force the host to wait until all queued
operations have completed and any errors have been reported.  Only after the
synchronization do we reset the device, ensuring a clean teardown and reliable
error reporting.  This pattern is considered good practice for robust CUDA
applications.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Utility macro for error checking */
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* Simple kernel that increments each element of an array */
__global__ void addOne(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] += 1;
    }
}

int main(void) {
    const int N = 1 << 20;  // Number of elements
    size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_arr = (int*)malloc(size);
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i) h_arr[i] = i;

    // Allocate device memory
    int *d_arr = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_arr, size));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addOne<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // **Good practice**: synchronize before resetting the device
    // Ensures all GPU work is finished and errors are reported
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    for (int i = 0; i < 5; ++i) {
        printf("h_arr[%d] = %d\n", i, h_arr[i]);
    }

    // Clean up device memory
    CHECK_CUDA(cudaFree(d_arr));

    // Reset device (good practice after synchronization)
    CHECK_CUDA(cudaDeviceReset());

    // Free host memory
    free(h_arr);

    printf("Program completed successfully.\n");
    return EXIT_SUCCESS;
}
