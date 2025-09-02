/*
Implement a kernel that copies one device array to another: b[i] = a[i].

My thinking: The goal is to write a simple CUDA C program that demonstrates how to copy data between two arrays residing entirely on the GPU.  
- We will allocate two device arrays `d_a` and `d_b` of the same length.  
- A host array `h_a` will be initialized with some values, transferred to `d_a`, and then a CUDA kernel will run to copy each element from `d_a` to `d_b`.  
- After the kernel finishes, `d_b` is copied back to a host array `h_b` for verification.  
- Error checking is performed after each CUDA call to make debugging easier.  
- The kernel will use a one‑dimensional grid and block configuration, with each thread copying one element.  
- The program prints the first few elements of the source and destination arrays to verify the copy.  
- All allocations and deallocations are done with proper cleanup.  
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple CUDA error checking macro
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that copies array a to array b: b[i] = a[i]
__global__ void copyKernel(const float *a, float *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        b[idx] = a[idx];
    }
}

int main(void) {
    const size_t N = 1 << 20; // 1M elements
    const size_t SIZE = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(SIZE);
    float *h_b = (float*)malloc(SIZE);
    if (!h_a || !h_b) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array with some values
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i) * 0.5f;
    }

    // Allocate device memory
    float *d_a = nullptr;
    float *d_b = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_a, SIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_b, SIZE));

    // Copy host array to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, SIZE, cudaMemcpyHostToDevice));

    // Launch kernel: 256 threads per block
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    copyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_b, d_b, SIZE, cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("First 10 elements of source (a):\n");
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_a[i]);
    }
    printf("\n");

    printf("First 10 elements of destination (b):\n");
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_b[i]);
    }
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    free(h_a);
    free(h_b);

    return 0;
}
