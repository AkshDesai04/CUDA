```cuda
/*
Pass the scalar `a` via constant memory instead of as a kernel argument.

Thinking:
1. The goal is to demonstrate how to use CUDA constant memory to store a scalar
   value that can be read by all threads in a kernel without being passed as a
   kernel argument. Constant memory is cached and ideal for values that are
   read-only and the same for all threads.
2. We will:
   - Declare a __constant__ variable on the device to hold the scalar.
   - Copy the scalar value from host to this constant memory using
     cudaMemcpyToSymbol.
   - Write a simple kernel that multiplies each element of an array by the
     scalar read from constant memory.
   - Perform allocation, initialization, data transfer, kernel launch, and
     result retrieval on the host.
3. The code will include basic error checking macros and will print the
   resulting array to verify correctness.
4. No kernel argument will be used for the scalar; instead, it will be accessed
   directly from the constant memory symbol within the kernel.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 10  // Size of the array
#define THREADS_PER_BLOCK 256

// Constant memory declaration
__constant__ float const_a;

// CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel that multiplies each element by the scalar in constant memory
__global__ void multiplyKernel(float *d_out, const float *d_in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_out[idx] = d_in[idx] * const_a;
    }
}

int main(void) {
    float host_in[N], host_out[N];
    float a = 2.5f;  // Scalar to multiply with

    // Initialize input array
    for (int i = 0; i < N; ++i) {
        host_in[i] = (float)i;
    }

    // Allocate device memory
    float *d_in = NULL;
    float *d_out = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, N * sizeof(float)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_in, host_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // Copy scalar 'a' to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(const_a, &a, sizeof(float)));

    // Launch kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    multiplyKernel<<<blocks, THREADS_PER_BLOCK>>>(d_out, d_in, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(host_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    printf("Input array:\n");
    for (int i = 0; i < N; ++i) {
        printf("%f ", host_in[i]);
    }
    printf("\n\nOutput array (multiplied by %f):\n", a);
    for (int i = 0; i < N; ++i) {
        printf("%f ", host_out[i]);
    }
    printf("\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
```