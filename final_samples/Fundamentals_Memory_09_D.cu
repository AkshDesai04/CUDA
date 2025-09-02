```cuda
/*
What is the downside of using pinned memory? (It's a limited resource and reduces memory available to the OS and other applications).
I decided to write a small CUDA program that demonstrates the allocation of pinned (page‑locked) host memory,
uses it for a simple vector addition kernel, and then reports the amount of free device memory before and
after the allocation.  The code is deliberately straightforward so it can be compiled with any CUDA
toolkit.  The goal is to illustrate that pinned memory is a scarce resource on the host side; the
allocation succeeds but may leave less memory for the operating system and other applications.
The program also highlights that freeing the pinned memory restores the host memory pool, but the
GPU side still keeps the pinned area reserved for DMA.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

// Simple vector addition kernel
__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

// Error checking macro
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

int main(void)
{
    size_t size = N * sizeof(float);
    float *h_a, *h_b, *h_c;          // Host pointers
    float *d_a, *d_b, *d_c;          // Device pointers

    // Print device free memory before pinned allocation
    size_t freeMem, totalMem;
    CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
    printf("Before pinned allocation: Free device memory = %zu bytes\n", freeMem);

    // Allocate pinned (page-locked) host memory
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
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify result
    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5) {
            printf("Mismatch at index %d: got %f, expected %f\n", i, h_c[i], expected);
            success = false;
            break;
        }
    }
    if (success)
        printf("Vector addition succeeded.\n");

    // Free device memory
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    // Print device free memory after freeing device allocations but before freeing pinned memory
    CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
    printf("After freeing device memory but before freeing pinned memory: Free device memory = %zu bytes\n", freeMem);

    // Free pinned host memory
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_c));

    // Print device free memory after freeing all allocations
    CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
    printf("After freeing all memory: Free device memory = %zu bytes\n", freeMem);

    return 0;
}
```