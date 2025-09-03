/*
Aim of the program:
Pass the scalar `a` via constant memory instead of as a kernel argument.

Thought process:
- The requirement is to demonstrate how to use CUDA constant memory to store a scalar value that will be used by a kernel.
- Constant memory is read‑only for the kernel and cached on the device, so it is efficient for broadcasting a single value to many threads.
- We'll allocate a constant memory variable: `__constant__ float d_a;`.
- On the host side, we will copy the value of `a` into this constant symbol using `cudaMemcpyToSymbol`.
- The kernel will read `d_a` directly from constant memory, avoiding passing it as an argument.
- For illustration, the kernel will compute an array where each element `i` is set to `d_a * i`.
- We'll also include simple error checking and a verification step on the host.
- The program is self‑contained, compiles with `nvcc`, and writes the result to a file or prints a few elements to confirm correctness.

*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Constant memory declaration (scalar a)
__constant__ float d_a;

// Kernel that multiplies each index by constant a and stores result
__global__ void scaleByConstant(float *out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = d_a * idx;
    }
}

// Helper macro for checking CUDA errors
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void)
{
    const int n = 1 << 20;           // 1M elements
    const size_t size = n * sizeof(float);
    const float host_a = 3.1415926535f;  // Example scalar

    // Allocate host memory
    float *h_out = (float *)malloc(size);
    if (!h_out) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    float *d_out = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_out, size));

    // Copy scalar a to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_a, &host_a, sizeof(float)));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    scaleByConstant<<<blocksPerGrid, threadsPerBlock>>>(d_out, n);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // Simple verification: print first 5 results
    printf("First 5 results (index * a):\n");
    for (int i = 0; i < 5; ++i) {
        printf("h_out[%d] = %f\n", i, h_out[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_out));
    free(h_out);

    return EXIT_SUCCESS;
}
