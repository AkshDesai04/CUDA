/*
Aim of the program: Launch with a grid of N blocks. Pass an integer array `int* out`. Have each thread write its `blockIdx.x` to `out[blockIdx.x]`.

Thinking:
1. We want a CUDA kernel that receives a pointer to an integer array (`int* out`) on the device.
2. For each thread, we simply read its `blockIdx.x` and write that value into the array at index `blockIdx.x`. 
   All threads in the same block will write the same value to the same location; since the value is identical, there is no
   race condition affecting correctness – the last write will just overwrite with the same value.
3. The grid size will be N blocks. The number of threads per block can be chosen arbitrarily; for illustration we use 256 threads per block,
   but the program will also work with a single thread per block.
4. In the host code we allocate host and device memory, launch the kernel with a 1-D grid of N blocks,
   copy the result back, and print the array to verify that each element holds its block index.
5. Basic CUDA error checking is included for robustness.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 16                    // Number of blocks in the grid
#define THREADS_PER_BLOCK 256   // Number of threads per block (can be changed)

// Helper macro for checking CUDA errors
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Kernel that writes blockIdx.x to out[blockIdx.x]
__global__ void writeBlockIdx(int *out)
{
    // Each thread writes the same value: the index of its block
    out[blockIdx.x] = blockIdx.x;
}

int main(void)
{
    int *h_out = NULL;      // Host array
    int *d_out = NULL;      // Device array

    size_t size = N * sizeof(int);

    // Allocate host memory
    h_out = (int*)malloc(size);
    if (h_out == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    cudaCheckError(cudaMalloc((void**)&d_out, size));

    // Launch kernel: N blocks, THREADS_PER_BLOCK threads per block
    writeBlockIdx<<<N, THREADS_PER_BLOCK>>>(d_out);
    cudaCheckError(cudaGetLastError()); // Check for launch errors

    // Wait for GPU to finish
    cudaCheckError(cudaDeviceSynchronize());

    // Copy result back to host
    cudaCheckError(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // Print the results
    printf("Result from GPU:\n");
    for (int i = 0; i < N; ++i) {
        printf("out[%d] = %d\n", i, h_out[i]);
    }

    // Clean up
    cudaFree(d_out);
    free(h_out);

    return 0;
}
