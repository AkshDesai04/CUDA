/*
Place your __device__ function definition after the __global__ kernel that calls it.
Does it compile? (It might require a forward declaration).

Thinking:
  In CUDA, a __device__ function must be declared before it is used
  unless a forward declaration is provided. If we define the __device__
  function after the __global__ kernel that calls it, the compiler
  will complain because it hasn't seen the function definition yet.
  To make it compile, we provide a prototype (forward declaration)
  of the device function before the kernel. Then we define the
  function after the kernel. This example demonstrates that
  approach: a simple device function that adds two integers,
  a kernel that uses it, and a host main that runs the kernel.
  The code compiles with a standard CUDA compiler (nvcc).
*/

#include <cstdio>
#include <cuda_runtime.h>

// Forward declaration of the __device__ function
__device__ int add(int a, int b);

// __global__ kernel that calls the __device__ function
__global__ void vectorAddKernel(const int* a, const int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Call the device function defined later
        c[idx] = add(a[idx], b[idx]);
    }
}

// Definition of the __device__ function (placed after the kernel)
__device__ int add(int a, int b) {
    return a + b;
}

// Helper macro for checking CUDA errors
#define CUDA_CHECK(err)                                            \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA Error: %s (err_num=%d)\n",            \
                cudaGetErrorString(err), (int)(err));              \
        exit(EXIT_FAILURE);                                        \
    }

int main() {
    const int N = 10;
    int h_a[N], h_b[N], h_c[N];

    // Initialize input arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    int *d_a, *d_b, *d_c;
    size_t size = N * sizeof(int);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_a, size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size));
    CUDA_CHECK(cudaMalloc((void**)&d_c, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Print results
    printf("Result of vector addition (a + b):\n");
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
