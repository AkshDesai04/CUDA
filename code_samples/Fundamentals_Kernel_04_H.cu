```cuda
/*
Aim: Launch with a grid of N blocks. Pass an integer array `int* out`.  
Have each thread write its `blockIdx.x` to `out[blockIdx.x]`.

Thinking:
1. We need a kernel that is launched with a grid of N blocks.  
2. Each thread within a block should write the value of its block index
   (`blockIdx.x`) to a location in the output array that corresponds to
   that block (`out[blockIdx.x]`).  All threads in the same block write
   the same value to the same memory location; this is allowed and will
   result in the final value being that block index.  
3. The host code will:
   - Parse the number of blocks N from the command line.
   - Allocate an array of N integers on the host and on the device.
   - Launch the kernel with `gridDim.x = N` and some reasonable number
     of threads per block (e.g., 256).
   - Copy the result back to the host and print it.
4. We guard against out‑of‑bounds accesses by checking that
   `blockIdx.x < N` before writing.
5. Since every thread writes the same value, we don't need any
   synchronization inside the block; the last write will leave the
   value unchanged.
6. The code is fully self‑contained and can be compiled with `nvcc`.

*/

#include <stdio.h>
#include <stdlib.h>

__global__ void write_block_indices(int *out, int N)
{
    // Each thread writes blockIdx.x to out[blockIdx.x] if within bounds
    if (blockIdx.x < N)
    {
        out[blockIdx.x] = blockIdx.x;
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <N_blocks>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int N = atoi(argv[1]); // number of blocks
    if (N <= 0)
    {
        fprintf(stderr, "Error: N must be positive.\n");
        return EXIT_FAILURE;
    }

    // Allocate host array
    int *h_out = (int *)malloc(N * sizeof(int));
    if (!h_out)
    {
        fprintf(stderr, "Host allocation failed.\n");
        return EXIT_FAILURE;
    }

    // Allocate device array
    int *d_out;
    cudaError_t err = cudaMalloc((void **)&d_out, N * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device allocation failed: %s\n", cudaGetErrorString(err));
        free(h_out);
        return EXIT_FAILURE;
    }

    // Launch kernel: N blocks, 256 threads per block
    const int threadsPerBlock = 256;
    dim3 gridDim(N);
    dim3 blockDim(threadsPerBlock);

    write_block_indices<<<gridDim, blockDim>>>(d_out, N);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return EXIT_FAILURE;
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    err = cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Memcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return EXIT_FAILURE;
    }

    // Print output
    printf("Output array (each element should equal its index):\n");
    for (int i = 0; i < N; ++i)
    {
        printf("out[%d] = %d\n", i, h_out[i]);
    }

    // Clean up
    cudaFree(d_out);
    free(h_out);

    return EXIT_SUCCESS;
}
```