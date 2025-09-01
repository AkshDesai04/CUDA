```cuda
/*
Remember to use `cudaFreeHost` to free memory allocated with `cudaMallocHost`.

Thinking:
The aim of this program is to demonstrate the proper use of cudaMallocHost
for pinned (page-locked) host memory and cudaFreeHost for freeing that
memory. We will perform a simple vector addition on the GPU. The steps
are:
1. Allocate pinned host memory for three arrays: A, B, and C.
2. Allocate device memory for the same arrays.
3. Copy input data from pinned host memory to device memory.
4. Launch a kernel to add vectors A and B into C.
5. Copy the result back from device to pinned host memory.
6. Print a few results to verify correctness.
7. Free all device memory with cudaFree.
8. Free the pinned host memory with cudaFreeHost.

Pinned host memory allows faster transfer between host and device,
but must be freed with cudaFreeHost. The program includes basic
error checking for CUDA calls and uses a small vector size for
simplicity.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024          // Number of elements
#define THREADS_PER_BLOCK 256

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                   \
    do {                                                   \
        cudaError_t err = call;                            \
        if (err != cudaSuccess) {                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", \
                    #call, __FILE__, __LINE__,             \
                    cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                            \
        }                                                  \
    } while (0)

int main()
{
    float *h_a = NULL; // Pinned host memory for A
    float *h_b = NULL; // Pinned host memory for B
    float *h_c = NULL; // Pinned host memory for C (result)

    float *d_a = NULL; // Device memory for A
    float *d_b = NULL; // Device memory for B
    float *d_c = NULL; // Device memory for C

    size_t size = N * sizeof(float);

    // Allocate pinned host memory
    CHECK_CUDA(cudaMallocHost((void**)&h_a, size));
    CHECK_CUDA(cudaMallocHost((void**)&h_b, size));
    CHECK_CUDA(cudaMallocHost((void**)&h_c, size));

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_a, size));
    CHECK_CUDA(cudaMalloc((void**)&d_b, size));
    CHECK_CUDA(cudaMalloc((void**)&d_c, size));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Print first 10 results to verify
    printf("First 10 results of A + B:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_c[%d] = %f\n", i, h_c[i]);
    }

    // Free device memory
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    // Free pinned host memory
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_c));

    printf("Program completed successfully.\n");
    return 0;
}
```