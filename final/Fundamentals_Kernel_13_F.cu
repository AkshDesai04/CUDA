/*
Implement element-wise multiplication: `c[i] = a[i] * b[i]`.

Thought process:

1. The goal is to perform a simple element-wise multiplication of two arrays `a` and `b`, storing the result in array `c`. 
2. In CUDA, this is a typical vector operation, so we will write a kernel that processes each element independently.
3. We need to allocate memory on both the host (CPU) and the device (GPU), transfer data to the GPU, execute the kernel, and then copy the results back.
4. A simple error-checking macro (`CUDA_CHECK`) will be used to wrap CUDA API calls for robustness.
5. For demonstration purposes, we will use a fixed array size (`N = 1024`) and initialize the arrays with deterministic values so that the result can be verified easily.
6. The kernel will calculate the global thread index and perform the multiplication only if the index is within bounds.
7. We will launch the kernel with a reasonable block size (256 threads per block) and calculate the required number of blocks.
8. After the kernel execution, we will copy the result back to the host, print a few entries to confirm correctness, and clean up all allocated memory.

This structure ensures clarity, proper memory management, and straightforward verification of the element-wise multiplication operation on the GPU.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",           \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Kernel to perform element-wise multiplication: c[i] = a[i] * b[i]
__global__ void vector_mul(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

int main(void) {
    const int N = 1024;          // Number of elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;           // Example: a[i] = i
        h_b[i] = 2.0f;               // Example: b[i] = 2.0
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size));
    CUDA_CHECK(cudaMalloc((void**)&d_c, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    vector_mul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());  // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("First 10 results of c[i] = a[i] * b[i]:\n");
    for (int i = 0; i < 10; ++i) {
        printf("c[%d] = %f\n", i, h_c[i]);
    }

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
