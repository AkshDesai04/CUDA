/*
Aim: Instead of `exit(-1)`, use `cudaDeviceReset()` before exiting to clean up the context.

Thinking:
- The task requires a CUDA C program that demonstrates proper cleanup of the device context by calling `cudaDeviceReset()` instead of abruptly exiting with `exit(-1)`.
- We'll create a simple CUDA example (vector addition) to illustrate kernel launch and error handling.
- When an error occurs, the program will:
  1. Print an error message.
  2. Call `cudaDeviceReset()` to release resources and reset the device.
  3. Exit with a non‑zero status.
- The code will use a helper macro `CHECK_CUDA(call)` to simplify error checking.
- We ensure the program compiles as a `.cu` file and includes all necessary headers and device/kernel definitions.
- No extra text or files are produced; only the single `.cu` source code is output.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            /* Reset device before exiting to clean up the context */  \
            cudaDeviceReset();                                         \
            exit(1);                                                   \
        }                                                              \
    } while (0)

// Simple vector addition kernel
__global__ void vecAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Host memory allocation failed\n");
        /* No device to reset here, just exit */
        exit(1);
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void**)&d_a, size));
    CHECK_CUDA(cudaMalloc((void**)&d_b, size));
    CHECK_CUDA(cudaMalloc((void**)&d_c, size));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError()); // Check kernel launch

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify results (simple check)
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            fprintf(stderr, "Verification failed at index %d: %f != %f\n",
                    i, h_c[i], h_a[i] + h_b[i]);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Vector addition successful, all values verified.\n");
    } else {
        /* Reset device before exiting to clean up the context */
        cudaDeviceReset();
        exit(1);
    }

    // Clean up device memory
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    // Clean up host memory
    free(h_a);
    free(h_b);
    free(h_c);

    // Reset device to clean up the context before normal exit
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
