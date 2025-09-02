/*
Integrate a CUB block-level primitive, like BlockScan, into your own custom kernel to perform a scan within a single thread block.

Thinking:
The goal is to demonstrate how to embed a CUB block‑level scan (BlockScan) inside a user defined CUDA kernel.  CUB provides a highly optimized, templated block‑scan that requires a small amount of shared memory (TempStorage).  In the kernel we declare the shared TempStorage, instantiate a BlockScan object, and invoke InclusiveSum (or ExclusiveSum) on each thread’s value.  The result is written back to global memory.  On the host side we allocate a small input array, launch the kernel with a single block (or multiple blocks) and copy back the results for verification.  The example uses integers and performs an inclusive scan of an array of ones to show the cumulative sum.

We keep the code self‑contained: includes, definitions, kernel, host code, and a small output for verification.
*/

#include <cstdio>
#include <cub/cub.cuh>

constexpr int BLOCK_SIZE = 256; // number of threads per block

// Kernel that performs a block‑level inclusive scan using CUB's BlockScan
__global__ void blockScanKernel(const int *d_in, int *d_out, int N)
{
    // Temporary storage required by CUB BlockScan
    __shared__ cub::BlockScan<int, BLOCK_SIZE>::TempStorage temp_storage;
    // BlockScan object
    cub::BlockScan<int, BLOCK_SIZE> block_scan;

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (gid < N) ? d_in[gid] : 0; // guard against out of bounds
    int out;

    // Perform inclusive sum within the block
    block_scan.InclusiveSum(val, out, cub::Sum(), temp_storage);

    // Write result back to global memory
    if (gid < N) d_out[gid] = out;
}

int main()
{
    const int N = 1024; // total number of elements (multiple of BLOCK_SIZE for simplicity)

    // Host allocation
    int *h_in  = new int[N];
    int *h_out = new int[N];
    for (int i = 0; i < N; ++i) h_in[i] = 1; // simple input: all ones

    // Device allocation
    int *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));

    // Copy input to device
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch configuration
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    blockScanKernel<<<gridDim, blockDim>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Simple verification: print first 10 results
    printf("Inclusive scan results (first 10 elements):\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}
