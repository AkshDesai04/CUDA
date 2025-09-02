/*
Aim: Write a kernel to shift (rotate) an array by `k` positions using shared memory.

Thoughts:
1. The rotation (shift) operation is a global transformation: each element at index i in the input array
   should move to index (i + k) % N in the output array.
2. To satisfy the requirement of using shared memory, we will load the entire array (or a block
   of the array) into shared memory before performing the write.
3. The simplest way to demonstrate shared memory usage is to launch one block that covers the whole
   array.  Each thread copies its element from global memory to shared memory, synchronizes,
   then writes the element to its rotated position in the output array.
4. For larger arrays a more sophisticated approach would be needed (e.g., using multiple blocks
   and handling wrap‑around), but for clarity and brevity we restrict the example to arrays that
   fit into a single block and into the shared memory budget.
5. The kernel will therefore:
   - Compute global thread index (tid).
   - Load input[tid] into shared memory s[tid] if tid < N.
   - Synchronize to ensure all loads are complete.
   - Compute newIdx = (tid + k) % N and write s[tid] to output[newIdx].
6. On the host side, we allocate device buffers, copy the input array, launch the kernel with
   one block of N threads, copy back the output array, and print both arrays.
7. We add basic error checking and handle the case where N is larger than the maximum number of
   threads per block by exiting with an error message.
8. The shared memory size requested for the kernel launch is N * sizeof(int), which is
   acceptable for moderate array sizes (e.g., up to several thousand elements).
9. All code is written in plain C/C++ CUDA style and is self‑contained in a single .cu file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",    \
                    __func__, __FILE__, __LINE__,                \
                    cudaGetErrorString(err));                    \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

__global__ void shiftKernel(const int *in, int *out, int N, int k)
{
    extern __shared__ int s[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        // Load into shared memory
        s[tid] = in[tid];
    }
    __syncthreads();

    if (tid < N) {
        int newIdx = (tid + k) % N;
        out[newIdx] = s[tid];
    }
}

int main(void)
{
    // Example array
    int N = 12;                     // Size of the array
    int k = 3;                      // Number of positions to shift
    int h_in[N];
    int h_out[N];

    // Initialize input array
    for (int i = 0; i < N; ++i) {
        h_in[i] = i + 1;  // 1, 2, ..., N
    }

    // Device memory allocation
    int *d_in = NULL;
    int *d_out = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_in, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_out, N * sizeof(int)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    // Determine block and grid dimensions
    int threadsPerBlock = N; // one block covering the whole array
    int blocks = 1;

    // Ensure we do not exceed the maximum threads per block
    int maxThreadsPerBlock;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    maxThreadsPerBlock = prop.maxThreadsPerBlock;
    if (threadsPerBlock > maxThreadsPerBlock) {
        fprintf(stderr, "Error: Number of threads (%d) exceeds maximum per block (%d).\n",
                threadsPerBlock, maxThreadsPerBlock);
        return EXIT_FAILURE;
    }

    size_t sharedMemSize = N * sizeof(int);

    // Launch kernel
    shiftKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_in, d_out, N, k);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Input array:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_in[i]);
    }
    printf("\n");

    printf("Shifted array (k = %d):\n", k);
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    // Clean up
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}
