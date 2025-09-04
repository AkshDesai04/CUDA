/*
Initialize the bin array to all zeros using `cudaMemset` before launching the kernel.

Thinking:
The goal is to demonstrate the correct usage of `cudaMemset` to zero out a device array
before a kernel uses it.  I will write a small CUDA program that:
1. Allocates a device array called `d_bin` that holds N integers.
2. Calls `cudaMemset(d_bin, 0, N * sizeof(int))` to set every element to zero.
3. Launches a kernel that writes a simple pattern into the bin array (for example, 
   setting each element to its index).  This confirms that the array was indeed
   initialized to zero, as the kernel would otherwise operate on garbage data.
4. Copies the result back to host memory and prints it.
The program includes basic error checking, uses `cudaGetLastError` after the kernel launch,
and prints the final bin contents.  This illustrates the standard workflow:
allocate -> memset -> kernel -> copy -> verify.  The code is fully self-contained
and can be compiled with `nvcc` to produce a .cu file. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10  // Size of the bin array

// Kernel that writes its thread index into the bin array
__global__ void fillBins(int *bin, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        bin[idx] = idx;  // Simple pattern: each bin holds its index
    }
}

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main(void)
{
    int *d_bin = NULL;
    int host_bin[N];

    // Allocate device memory for the bin array
    CHECK_CUDA(cudaMalloc((void **)&d_bin, N * sizeof(int)));

    // Initialize the bin array to zeros using cudaMemset
    CHECK_CUDA(cudaMemset(d_bin, 0, N * sizeof(int)));

    // Launch the kernel with enough threads to cover N elements
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    fillBins<<<blocks, threadsPerBlock>>>(d_bin, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy the result back to host memory
    CHECK_CUDA(cudaMemcpy(host_bin, d_bin, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print the bin contents
    printf("Bin array after kernel execution:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("bin[%d] = %d\n", i, host_bin[i]);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_bin));

    return 0;
}
