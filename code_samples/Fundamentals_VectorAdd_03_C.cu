```cuda
/*
 * Aim: Explain in a comment why the grid calculation `(N + BLOCK_SIZE - 1) / BLOCK_SIZE`
 * launches enough blocks to cover all elements.
 *
 * Thinking:
 *
 * The goal of a CUDA kernel launch is to ensure that every element of an array
 * (or each iteration of a loop) has a unique thread that processes it.  Threads
 * are organized into blocks, and blocks are organized into a grid.  If we have
 * `N` elements and a block size of `BLOCK_SIZE`, we need at least
 * `ceil(N / BLOCK_SIZE)` blocks so that the product
 * (`number_of_blocks * BLOCK_SIZE`) is not smaller than `N`.
 *
 * Integer division in C/C++ truncates toward zero, so `N / BLOCK_SIZE`
 * gives us the floor of the exact division.  To obtain the ceiling value,
 * we add `BLOCK_SIZE - 1` to `N` before dividing:
 *
 *     ceil(N / BLOCK_SIZE)  ==  (N + BLOCK_SIZE - 1) / BLOCK_SIZE
 *
 * Why does this work?
 *
 * 1. If `N` is an exact multiple of `BLOCK_SIZE` (e.g. N = 32, BLOCK_SIZE = 8),
 *    then `N + BLOCK_SIZE - 1` equals `N + (BLOCK_SIZE - 1)` which is still
 *    less than the next multiple of BLOCK_SIZE.  Division truncates to `N /
 *    BLOCK_SIZE`, the correct number of blocks.
 *
 * 2. If `N` is not a multiple (e.g. N = 35, BLOCK_SIZE = 8), adding
 *    `BLOCK_SIZE - 1` (which is 7) gives 42.  Dividing by 8 yields 5, which is
 *    the smallest integer greater than 35/8 (which is 4.375).  Thus we get one
 *    extra block to cover the remaining 3 elements.
 *
 * The kernel typically uses an index calculation such as:
 *
 *     int idx = blockIdx.x * blockDim.x + threadIdx.x;
 *     if (idx < N) { /* process element idx */ }
 *
 * The guard `if (idx < N)` ensures that any threads belonging to the last
 * (possibly partially full) block that would access out‑of‑range memory are
 * safely ignored.  Consequently, launching `(N + BLOCK_SIZE - 1) / BLOCK_SIZE`
 * blocks guarantees that every element has a dedicated thread while not
 * allocating unnecessary threads beyond the required count.
 *
 * This simple arithmetic trick is widely used in CUDA programming to compute
 * the minimal number of blocks needed for a given problem size.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256          // Number of threads per block
#define N         1000          // Number of elements

// Simple kernel that initializes an array with its index
__global__ void init_array(int *d_arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] = idx;
    }
}

int main(void)
{
    int *h_arr = (int*)malloc(N * sizeof(int));
    int *d_arr;
    cudaMalloc((void**)&d_arr, N * sizeof(int));

    // Compute number of blocks needed to cover all N elements
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch kernel
    init_array<<<numBlocks, BLOCK_SIZE>>>(d_arr, N);

    // Copy result back to host
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify a few elements
    for (int i = 0; i < 10; ++i) {
        printf("h_arr[%d] = %d\n", i, h_arr[i]);
    }

    // Clean up
    cudaFree(d_arr);
    free(h_arr);
    return 0;
}
```