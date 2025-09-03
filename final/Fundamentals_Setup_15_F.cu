/*
Call `cudaDeviceSynchronize()` before `cudaDeviceReset()` and explain why this can be good practice.
This program demonstrates a simple CUDA workflow that launches a kernel, waits for all device
operations to finish using `cudaDeviceSynchronize()`, and then resets the device with
`cudaDeviceReset()`. The reason for the synchronization step is that `cudaDeviceReset()` forces
the device to release all resources and abort any pending work. If there are unfinished kernels
or memory operations, resetting the device prematurely can lead to incomplete work, corrupted
results, or even crashes. By synchronizing first, we guarantee that all GPU tasks have completed
and that the host and device are in a clean, known state before the reset occurs. This pattern
is especially useful in long-running or interactive applications where device resets may be
performed repeatedly, ensuring robust and predictable behavior.

The thinking behind this approach is:
- After launching kernels, the host thread continues executing without waiting for the GPU.
- If we reset the device immediately, the GPU may still be executing or in the middle of memory
  copies, leading to undefined behavior.
- Synchronizing forces the host to wait until all device work is finished.
- Once synchronized, a reset safely cleans up all allocations and resets the device state.
- This guarantees that subsequent operations start from a clean slate and that any resources
  (like memory or contexts) are properly released.

Below is a minimal example that follows this best practice.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_ERROR(call)                                           \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",        \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Simple kernel that adds two arrays element-wise
__global__ void addArrays(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = i * 0.5f;
        h_b[i] = i * 1.5f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_ERROR(cudaMalloc((void**)&d_a, size));
    CHECK_ERROR(cudaMalloc((void**)&d_b, size));
    CHECK_ERROR(cudaMalloc((void**)&d_c, size));

    // Copy data from host to device
    CHECK_ERROR(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addArrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_ERROR(cudaGetLastError()); // Check for launch errors

    // Synchronize to ensure kernel completion before reset
    CHECK_ERROR(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_ERROR(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    for (int i = 0; i < 5; ++i) {
        printf("h_c[%d] = %f\n", i, h_c[i]);
    }

    // Reset the device after all operations are complete
    CHECK_ERROR(cudaDeviceReset());

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
