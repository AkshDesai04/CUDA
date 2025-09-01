/*
Aim of the program: 
Launch with a grid of N blocks. Pass an integer array `int* out`. Have each thread write its `blockIdx.x` to `out[blockIdx.x]`.

Thinking:
- The requirement is to write the block index into a device array. Each block writes to its own element, indexed by blockIdx.x. 
- The simplest way is to launch one thread per block (i.e., threadsPerBlock = 1). In that case, each block writes exactly once. 
- Even if we launch more than one thread per block, every thread will write the same value to the same element `out[blockIdx.x]`. That is harmless because all threads write the same value; the final value will be that block index. 
- For clarity and to avoid unnecessary race conditions, we choose one thread per block. 
- Error checking is added to ensure CUDA calls succeed. 
- The host program accepts an optional command line argument for the number of blocks N; if not provided, defaults to 10. 
- After kernel execution, the device array is copied back to host and printed. 
- The code is self‑contained in a single .cu file and can be compiled with nvcc.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void writeBlockIdx(int *out)
{
    // Each block writes its blockIdx.x into out[blockIdx.x].
    // Using one thread per block (blockDim.x == 1) to avoid races.
    int idx = blockIdx.x;
    out[idx] = idx;
}

int main(int argc, char *argv[])
{
    int N = 10; // default number of blocks
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "Invalid number of blocks: %d\n", N);
            return EXIT_FAILURE;
        }
    }

    // Allocate host memory
    int *h_out = (int *)malloc(N * sizeof(int));
    if (!h_out) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    int *d_out = NULL;
    cudaError_t err = cudaMalloc((void **)&d_out, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_out);
        return EXIT_FAILURE;
    }

    // Launch kernel: one block per blockIdx.x, one thread per block
    writeBlockIdx<<<N, 1>>>(d_out);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return EXIT_FAILURE;
    }

    // Copy results back to host
    err = cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return EXIT_FAILURE;
    }

    // Print results
    printf("Output array (size %d):\n", N);
    for (int i = 0; i < N; ++i) {
        printf("out[%d] = %d\n", i, h_out[i]);
    }

    // Clean up
    cudaFree(d_out);
    free(h_out);

    return EXIT_SUCCESS;
}
