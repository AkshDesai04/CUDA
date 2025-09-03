/*
Write a `CHECK_KERNEL_LAUNCH()` macro that calls `cudaPeekAtLastError()` and `cudaDeviceSynchronize()` and reports any errors.

Thinking:
- The macro should encapsulate the common pattern of checking for kernel launch errors and device synchronization errors after launching a kernel.
- It will call `cudaPeekAtLastError()` to capture any error that occurred during the kernel launch (this does not block execution).
- It will then call `cudaDeviceSynchronize()` to block until the device has completed all preceding tasks and to catch any errors that occurred during kernel execution.
- If either call returns an error, the macro should print a descriptive message to `stderr` using `cudaGetErrorString()` to translate the error code.
- The macro will be defined using a `do { ... } while(0)` construct to ensure it behaves like a single statement.
- For demonstration, a simple vector addition kernel will be included, and the macro will be invoked after launching the kernel.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_KERNEL_LAUNCH() \
    do { \
        cudaError_t err = cudaPeekAtLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err)); \
        } \
        err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Device synchronization error: %s\n", cudaGetErrorString(err)); \
        } \
    } while (0)

// Simple vector addition kernel
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void) {
    const int N = 1 << 20;  // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_a[i] = i * 1.0f;
        h_b[i] = (N - i) * 1.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Check for errors
    CHECK_KERNEL_LAUNCH();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify a few results
    bool success = true;
    for (int i = 0; i < 10; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5) {
            fprintf(stderr, "Mismatch at %d: %f vs %f\n", i, h_c[i], expected);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Kernel executed successfully. Sample outputs verified.\n");
    }

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
