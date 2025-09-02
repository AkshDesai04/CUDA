```cuda
/* 
Can a kernel launch fail with `cudaErrorInvalidDevice`? (Unlikely, as the context is already established).

Thinking:
----------
The CUDA runtime manages a per-device context. A kernel launch requires a valid
context on the current device. Under normal circumstances, once a context has
been created (by calling any CUDA API that implicitly creates one, e.g. 
cudaMalloc or cudaSetDevice), a kernel launch should not return
`cudaErrorInvalidDevice`. However, if the context is destroyed (via
cudaDeviceReset) or if we attempt to launch a kernel on a device that
has no active context (or an invalid device ID), the launch will fail.
This program demonstrates two scenarios:

1. A normal kernel launch after a valid context has been established.
2. An attempt to launch a kernel after the context has been destroyed
   with cudaDeviceReset, which results in `cudaErrorInvalidDevice`.

The program prints the error string for each launch so we can see the
exact error returned by the runtime.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",        \
                    __FILE__, __LINE__, err, cudaGetErrorString(err));  \
            return -1;                                                  \
        }                                                               \
    } while (0)

// Simple kernel that adds two arrays element-wise
__global__ void addKernel(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}

int main(void)
{
    const int N = 1 << 20;          // 1M elements
    const int SIZE = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(SIZE);
    float *h_b = (float*)malloc(SIZE);
    float *h_c = (float*)malloc(SIZE);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_a, SIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_b, SIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_c, SIZE));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, SIZE, cudaMemcpyHostToDevice));

    // 1. Normal kernel launch
    printf("=== Normal kernel launch ===\n");
    dim3 threadsPerBlock(256);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    addKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed with error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel launch succeeded.\n");
    }

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, SIZE, cudaMemcpyDeviceToHost));

    // Verify a few results
    for (int i = 0; i < 5; ++i) {
        printf("h_c[%d] = %f\n", i, h_c[i]);
    }

    // 2. Kernel launch after destroying the context
    printf("\n=== Kernel launch after cudaDeviceReset ===\n");
    CUDA_CHECK(cudaDeviceReset());   // Destroys the current context

    // Attempt to launch kernel again on the same device pointers
    // Note: device pointers are no longer valid after cudaDeviceReset
    addKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch after reset failed with error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel launch after reset succeeded (unexpected).\n");
    }

    // Clean up host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```