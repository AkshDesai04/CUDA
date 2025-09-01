```cuda
/*
Aim of the program:
Generalize the kernel to work on arrays. It should take `int* a`, `int* b`, and `int N`. Each thread `i` should compute `b[i] = a[i] + 10;`.

Thinking and design:
- We need a CUDA kernel that accepts two integer pointers and the array length.
- The kernel will compute the global thread index and guard against out-of-bounds.
- In main we will allocate host arrays, initialize `a` with some data, allocate device memory, copy `a` to device, launch the kernel, copy back `b`, and print a few results to verify correctness.
- Error checking will be minimal but sufficient to catch obvious failures.
- The code will be self‑contained in a single .cu file, ready to compile with `nvcc`.
- We'll use a simple block size (e.g., 256) and compute the number of blocks accordingly.

*/

#include <stdio.h>
#include <cuda.h>

#define CHECK_CUDA(call)                                           \
    do {                                                            \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",      \
                    #call, __FILE__, __LINE__,                    \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                    \
        }                                                           \
    } while (0)

// Kernel: each thread adds 10 to one element of array a and stores result in b
__global__ void addTenKernel(const int* a, int* b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        b[idx] = a[idx] + 10;
    }
}

int main(void) {
    const int N = 1024;                     // Size of the arrays
    const int ARRAY_BYTES = N * sizeof(int);

    // Allocate host memory
    int *h_a = (int*)malloc(ARRAY_BYTES);
    int *h_b = (int*)malloc(ARRAY_BYTES);
    if (!h_a || !h_b) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host array a with some values
    for (int i = 0; i < N; ++i) {
        h_a[i] = i; // Example data: 0,1,2,...
    }

    // Allocate device memory
    int *d_a = NULL;
    int *d_b = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_a, ARRAY_BYTES));
    CHECK_CUDA(cudaMalloc((void**)&d_b, ARRAY_BYTES));

    // Copy input data from host to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice));

    // Launch kernel
    const int THREADS_PER_BLOCK = 256;
    const int NUM_BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    addTenKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_a, d_b, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());
    // Wait for GPU to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CCUDA(cudaMemcpy(h_b, d_b, ARRAY_BYTES, cudaMemcpyDeviceToHost));

    // Print a few results to verify
    printf("First 10 results (b[i] = a[i] + 10):\n");
    for (int i = 0; i < 10; ++i) {
        printf("b[%d] = %d\n", i, h_b[i]);
    }

    // Clean up
    free(h_a);
    free(h_b);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));

    printf("Program completed successfully.\n");
    return EXIT_SUCCESS;
}
```