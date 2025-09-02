/*
Aim: Use Nsight Compute to visualize the memory chart and see the inefficiency of the strided access.

Thinking:
- To demonstrate strided memory access inefficiency, I will implement a simple CUDA kernel that copies elements from a source array to a destination array using a stride that is larger than 1. This will cause each thread to access memory locations that are not contiguous, leading to poorly coalesced memory transactions.
- The kernel will compute an index based on the global thread ID and multiply it by the stride to determine which element to read from the source. The destination index will be the same as the global thread ID to keep writes contiguous.
- On the host side, I will allocate a source array of size N, initialize it with known values, and allocate a destination array of the same size. I will launch the kernel with a block size of 256 and a grid size that covers all elements. After the kernel execution, I will copy the result back to host memory and verify that the values were copied correctly.
- By running this program under Nsight Compute, the user can inspect the memory chart and observe the high memory transaction count and poor cache utilization for the strided access pattern.
- For comparison, I will also include a coalesced copy kernel that simply copies elements with no stride. The user can run both kernels and compare their memory charts to see the difference in efficiency.
- The program uses standard CUDA runtime API calls: cudaMalloc, cudaMemcpy, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, and cudaDeviceSynchronize. It also checks for CUDA errors for robustness.
- I will keep the program simple and self-contained, using only C and CUDA, so that it can be compiled with nvcc and run on any GPU that supports the CUDA version used.
- The code is written in a single .cu file, with the aim and thinking comment at the top as requested.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 20)          // Size of arrays (1M elements)
#define STRIDE 32            // Stride for strided access
#define BLOCK_SIZE 256

// Error checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel with strided source access
__global__ void strided_copy_kernel(const int *src, int *dst, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int src_idx = idx * stride;  // Access with stride
        if (src_idx < n) {
            dst[idx] = src[src_idx];
        } else {
            dst[idx] = -1; // Out-of-bounds guard
        }
    }
}

// Kernel with coalesced (non-strided) access
__global__ void coalesced_copy_kernel(const int *src, int *dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

int main() {
    // Allocate host memory
    int *h_src = (int *)malloc(N * sizeof(int));
    int *h_dst = (int *)malloc(N * sizeof(int));
    if (!h_src || !h_dst) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize source array
    for (int i = 0; i < N; ++i) {
        h_src[i] = i;
    }

    // Allocate device memory
    int *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc((void **)&d_src, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_dst, N * sizeof(int)));

    // Copy source to device
    CUDA_CHECK(cudaMemcpy(d_src, h_src, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch strided copy kernel
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("Launching strided_copy_kernel with stride %d\n", STRIDE);
    strided_copy_kernel<<<grid_size, BLOCK_SIZE>>>(d_src, d_dst, N, STRIDE);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("Verifying strided copy results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_dst[%d] = %d\n", i, h_dst[i]);
    }

    // Launch coalesced copy kernel for comparison
    printf("\nLaunching coalesced_copy_kernel\n");
    coalesced_copy_kernel<<<grid_size, BLOCK_SIZE>>>(d_src, d_dst, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("Verifying coalesced copy results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_dst[%d] = %d\n", i, h_dst[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    free(h_src);
    free(h_dst);

    return 0;
}
