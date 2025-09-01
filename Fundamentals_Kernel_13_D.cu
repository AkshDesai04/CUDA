```cpp
/*
Modify the kernel to take a fourth argument, `int scalar`, and compute `b[i] = a[i] + scalar;`.

Thinking:
1. The requirement is to adjust the existing CUDA kernel to accept an additional integer scalar parameter.
2. The kernel will compute the result for each element as `b[i] = a[i] + scalar`. 
3. Since `a[i]` is typically a floating point value, the scalar should be implicitly cast to float during addition.
4. The host code will allocate input array `a` and output array `b` on both host and device, 
   initialize `a`, and invoke the kernel with the scalar value.
5. Proper CUDA error checking is added for robustness.
6. The program uses simple command line arguments for size and scalar value to keep it flexible.
7. The kernel launch configuration uses a block size of 256 threads, and grid size computed from array length.
8. After kernel execution, the result array is copied back to host and printed.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    __FUNCTION__, __FILE__, __LINE__,            \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Kernel that adds a scalar to each element of array a and stores the result in b
__global__ void add_scalar(const float *a, float *b, int N, int scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        b[idx] = a[idx] + static_cast<float>(scalar);
    }
}

int main(int argc, char *argv[]) {
    // Default parameters
    int N = 1024;           // number of elements
    int scalar = 5;         // scalar to add

    // Override with command line arguments if provided
    if (argc >= 2) {
        N = atoi(argv[1]);
    }
    if (argc >= 3) {
        scalar = atoi(argv[2]);
    }

    size_t size = N * sizeof(float);
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);

    // Initialize input array
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_a = NULL;
    float *d_b = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_a, size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    add_scalar<<<gridSize, blockSize>>>(d_a, d_b, N, scalar);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to finish

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost));

    // Print first 10 results for verification
    printf("First 10 results of a[i] + %d:\n", scalar);
    for (int i = 0; i < 10 && i < N; ++i) {
        printf("a[%d] = %.2f, b[%d] = %.2f\n", i, h_a[i], i, h_b[i]);
    }

    // Cleanup
    free(h_a);
    free(h_b);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    return 0;
}
```