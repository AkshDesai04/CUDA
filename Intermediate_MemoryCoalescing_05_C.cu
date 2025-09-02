```cuda
/*
 * Aim of the program:
 * Implement the SoA version: create `float* d_xs, *d_ys, *d_zs`. 
 * Write a kernel that reads from `d_xs`: `output[i] = d_xs[i]`.
 *
 * Thinking:
 * 1. We need a simple CUDA program that demonstrates the Structure-of-Arrays (SoA) approach.
 * 2. Three device pointers `d_xs`, `d_ys`, `d_zs` will be allocated to hold separate arrays for x, y, and z components.
 * 3. The kernel will only read from `d_xs` and write each element into a separate `d_output` array.
 * 4. For demonstration, the host will create a small array of floats, copy it to `d_xs`, launch the kernel, and copy back the results.
 * 5. The program will print the original and copied arrays to verify correctness.
 * 6. Basic CUDA error checking macros are included for clarity.
 * 7. The program is self‑contained and can be compiled with `nvcc` as a single .cu file.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10
#define THREADS_PER_BLOCK 256

// CUDA error checking macro
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

// Kernel that copies elements from d_xs to d_output
__global__ void copyXsToOutput(const float* d_xs, float* d_output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_output[idx] = d_xs[idx];
    }
}

int main(void) {
    // Host arrays
    float h_xs[N], h_ys[N], h_zs[N];
    float h_output[N];

    // Initialize host data (only xs will be used in kernel)
    for (int i = 0; i < N; ++i) {
        h_xs[i] = (float)i + 0.5f;  // some arbitrary values
        h_ys[i] = 0.0f;              // not used
        h_zs[i] = 0.0f;              // not used
    }

    // Device pointers
    float *d_xs, *d_ys, *d_zs, *d_output;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_xs, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_ys, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_zs, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, N * sizeof(float)));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_xs, h_xs, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, h_ys, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_zs, h_zs, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    copyXsToOutput<<<blocks, THREADS_PER_BLOCK>>>(d_xs, d_output, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    printf("Original d_xs values:\n");
    for (int i = 0; i < N; ++i) {
        printf("%f ", h_xs[i]);
    }
    printf("\nCopied output values:\n");
    for (int i = 0; i < N; ++i) {
        printf("%f ", h_output[i]);
    }
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_zs));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
```