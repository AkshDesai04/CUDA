/*
Launch with a grid of N blocks. Pass an integer array `int* out`. Have each thread write its `blockIdx.x` to `out[blockIdx.x]`.

Thinking process:
1. Understand the requirement: we need a CUDA kernel that runs with a grid of N blocks. Each thread in each block writes the block's index (blockIdx.x) to an output array at the same index.
2. To avoid multiple writes to the same location, we will launch one thread per block (blockDim.x = 1). This guarantees that for each block there is exactly one write to `out[blockIdx.x]`.
3. The kernel will check that the block index is within bounds (bid < N) to be safe.
4. In the host code, we allocate a host array of size N, allocate device memory, launch the kernel, copy results back, print them, and free memory.
5. Use standard CUDA API: cudaMalloc, cudaMemcpy, cudaDeviceSynchronize, cudaGetLastError.
6. Ensure the code is self-contained, includes necessary headers, and compiles as a .cu file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void writeBlockIdx(int *out, int N)
{
    int bid = blockIdx.x;
    if (bid < N)
    {
        out[bid] = bid;
    }
}

int main(void)
{
    const int N = 256; // Example grid size; can be changed as needed
    int *h_out = (int *)malloc(N * sizeof(int));
    if (h_out == NULL)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    int *d_out = NULL;
    cudaError_t err = cudaMalloc((void **)&d_out, N * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_out);
        return EXIT_FAILURE;
    }

    // Launch one thread per block
    dim3 gridDim(N);
    dim3 blockDim(1);

    writeBlockIdx<<<gridDim, blockDim>>>(d_out, N);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return EXIT_FAILURE;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return EXIT_FAILURE;
    }

    // Copy back to host
    err = cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return EXIT_FAILURE;
    }

    // Print the results
    printf("Output array:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_out);
    free(h_out);

    return EXIT_SUCCESS;
}
