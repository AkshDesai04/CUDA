/*
Chain kernel calls. The output of the first kernel (b) becomes the input to a second kernel that, for example, squares every element.

Thinking:
1. Create a simple example where we have an input array `a`.
2. Kernel1: compute `b[i] = a[i] + 1`.
3. Kernel2: square each element of `b` in place: `b[i] *= b[i]`.
4. Allocate host and device memory, copy input to device, launch Kernel1, synchronize, launch Kernel2, synchronize.
5. Copy result back to host and print.
6. Include error checking and clean up resources.
7. Use simple grid/block configuration: 256 threads per block.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",              \
                    __FILE__, __LINE__, err, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel 1: compute b[i] = a[i] + 1
__global__ void kernel_add_one(const float* a, float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        b[idx] = a[idx] + 1.0f;
    }
}

// Kernel 2: square each element of b in place
__global__ void kernel_square(float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        b[idx] *= b[idx];
    }
}

int main(void) {
    const int N = 1024;
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    if (!h_a || !h_b) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
    }

    // Allocate device memory
    float *d_a = NULL, *d_b = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_a, size));
    CHECK_CUDA(cudaMalloc((void**)&d_b, size));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

    // Launch kernel_add_one
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    kernel_add_one<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Launch kernel_square
    kernel_square<<<blocksPerGrid, threadsPerBlock>>>(d_b, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost));

    // Print first 10 results for verification
    printf("First 10 results after chaining kernels:\n");
    for (int i = 0; i < 10; ++i) {
        printf("b[%d] = %f\n", i, h_b[i]);
    }

    // Cleanup
    free(h_a);
    free(h_b);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));

    return EXIT_SUCCESS;
}
