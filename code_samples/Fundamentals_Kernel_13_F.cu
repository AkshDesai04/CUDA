```cpp
/*
Aim of the program:
Implement element-wise multiplication: c[i] = a[i] * b[i]

Thinking process:
1. The goal is to create a CUDA program that multiplies two arrays element-wise.
2. We need a kernel that takes pointers to device arrays a, b, c and an integer N (number of elements).
3. Each thread will compute one element of c: c[threadIdx.x + blockIdx.x * blockDim.x] = a[...] * b[...].
4. We'll launch enough blocks so that all N elements are processed: (N + threadsPerBlock - 1) / threadsPerBlock blocks.
5. In main(), we'll:
   - Allocate host arrays a, b, c and fill a and b with some test data.
   - Allocate device arrays da, db, dc.
   - Copy a and b to device.
   - Launch kernel.
   - Copy result dc back to host array c.
   - Print the results for verification.
6. We'll include basic error checking for CUDA API calls for robustness.
7. We'll free device and host memory before exiting.
8. The program will compile with nvcc and produce an executable.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",        \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel for element-wise multiplication
__global__ void vectorMul(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] * b[idx];
    }
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays with sample data
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void**)&d_a, size));
    CHECK_CUDA(cudaMalloc((void**)&d_b, size));
    CHECK_CUDA(cudaMalloc((void**)&d_c, size));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorMul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("Sample results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("c[%d] = %f\n", i, h_c[i]);
    }

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return EXIT_SUCCESS;
}
```