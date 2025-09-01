/*
Aim: Use dim3 variables to define a grid of (4,1,1) and a block of (1,1,1) and launch the kernel.

Thinking:
- The purpose of this small CUDA program is to illustrate how to set up a grid and block
  configuration using the dim3 helper type.  The grid will have 4 blocks along the x‑dimension
  (and 1 in y and z), while each block will contain only a single thread.
- A simple kernel will be launched with these dimensions.  The kernel writes the block index
  to a pre‑allocated array so that the host can later read back and print the result.
- To keep the example minimal and self‑contained, the kernel will perform no complex
  computation, just a single store per thread.
- Host side code will allocate device memory, launch the kernel, copy the result back,
  and print the values.
- Error checking is added after each CUDA runtime call for robustness.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that stores the block index into the output array
__global__ void setBlockIdx(int *d_out)
{
    // Each block has only one thread (threadIdx.x == 0)
    int idx = blockIdx.x;          // block index along x
    d_out[idx] = idx;              // write to output
}

int main(void)
{
    const int N = 4;                     // Number of blocks = size of output array
    int h_out[N];                        // Host array to receive results

    int *d_out = NULL;
    cudaError_t err;

    // Allocate device memory
    err = cudaMalloc((void**)&d_out, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Define grid and block dimensions
    dim3 gridDim(4, 1, 1);   // (4,1,1)
    dim3 blockDim(1, 1, 1);  // (1,1,1)

    // Launch the kernel
    setBlockIdx<<<gridDim, blockDim>>>(d_out);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 1;
    }

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 1;
    }

    // Copy results back to host
    err = cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (device to host) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 1;
    }

    // Print results
    printf("Block indices written by kernel:\n");
    for (int i = 0; i < N; ++i) {
        printf("h_out[%d] = %d\n", i, h_out[i]);
    }

    // Clean up
    cudaFree(d_out);
    return 0;
}
