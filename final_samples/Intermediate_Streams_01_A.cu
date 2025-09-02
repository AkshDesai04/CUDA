```cuda
/*
Aim of the program: Use pinned host memory (`cudaMallocHost`) for your data buffers.

Thinking:
- The goal is to show how to allocate host memory that is page-locked (pinned) using cudaMallocHost.
- We'll create two input arrays and one output array on the host, all using pinned memory.
- Device memory will be allocated with cudaMalloc.
- We will copy data from host to device, run a simple element‑wise addition kernel,
  then copy the result back to host pinned memory.
- Pinned memory allows faster transfers and can be used with asynchronous copies; here we use
  synchronous copies for clarity but the pinned allocation remains.
- Proper error checking will be included, and all resources will be freed before exit.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024  // size of vectors

// Macro for error checking
#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.", \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Simple vector addition kernel
__global__ void vecAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main(void) {
    // Pinned host memory
    float *h_a = NULL;
    float *h_b = NULL;
    float *h_c = NULL;
    CHECK_CUDA(cudaMallocHost((void**)&h_a, N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost((void**)&h_b, N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost((void**)&h_c, N * sizeof(float)));

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Device memory
    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_b, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_c, N * sizeof(float)));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify some results
    printf("Sample results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_a[%d] + h_b[%d] = %f\n", i, i, h_c[i]);
    }

    // Free device memory
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    // Free pinned host memory
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_c));

    return 0;
}
```