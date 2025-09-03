/*
Explain in a comment why the grid calculation `(N + BLOCK_SIZE - 1) / BLOCK_SIZE` launches enough blocks to cover all elements.

The expression `(N + BLOCK_SIZE - 1) / BLOCK_SIZE` is a common idiom for computing the minimum number of blocks needed to cover `N` items when each block can process up to `BLOCK_SIZE` items. Here's the reasoning:

1. **Divide N by BLOCK_SIZE**  
   - If `N` is exactly divisible by `BLOCK_SIZE`, then `N / BLOCK_SIZE` gives the exact number of blocks required.  
   - If `N` is not divisible, integer division would floor the result, leaving some items unprocessed.

2. **Add (BLOCK_SIZE - 1)**  
   - By adding `BLOCK_SIZE - 1` before performing the integer division, we effectively round up to the next whole block when there's a remainder.  
   - For example, if `N = 10` and `BLOCK_SIZE = 4`, we have `10 + 3 = 13`. `13 / 4` (integer division) yields `3`, which is enough blocks: 3 blocks * 4 threads = 12 slots, covering all 10 items.

3. **Mathematical justification**  
   - Let `q = N / BLOCK_SIZE` (integer division) and `r = N % BLOCK_SIZE`.  
   - If `r == 0`, then `N + BLOCK_SIZE - 1 = q * BLOCK_SIZE + BLOCK_SIZE - 1`, and dividing by `BLOCK_SIZE` gives `q`.  
   - If `r > 0`, then `N + BLOCK_SIZE - 1 = q * BLOCK_SIZE + r + BLOCK_SIZE - 1 = (q + 1) * BLOCK_SIZE + (r - 1)`. Since `r - 1 < BLOCK_SIZE`, integer division yields `q + 1`. Thus we always get the ceiling of `N / BLOCK_SIZE`.

4. **Result**  
   - The computed grid size is the smallest integer greater than or equal to `N / BLOCK_SIZE`.  
   - Therefore, launching `gridSize` blocks ensures that there are at least `gridSize * BLOCK_SIZE` thread slots, which is enough to cover all `N` elements. If `N` is not a multiple of `BLOCK_SIZE`, the extra slots simply do not process any data (they can check bounds inside the kernel).

This formula is concise, portable, and guarantees full coverage without launching unnecessary blocks.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000          // Total number of elements
#define BLOCK_SIZE 256  // Threads per block

__global__ void init_array(int *arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = idx;  // Simple initialization
    }
}

int main(void)
{
    int *dev_arr;
    size_t bytes = N * sizeof(int);

    // Allocate device memory
    cudaMalloc((void**)&dev_arr, bytes);

    // Compute grid size using the formula
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch kernel
    init_array<<<gridSize, BLOCK_SIZE>>>(dev_arr, N);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Allocate host memory to retrieve results
    int *host_arr = (int*)malloc(bytes);
    cudaMemcpy(host_arr, dev_arr, bytes, cudaMemcpyDeviceToHost);

    // Verify a few elements
    printf("First 10 elements: ");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", host_arr[i]);
    }
    printf("\n");

    // Clean up
    free(host_arr);
    cudaFree(dev_arr);

    return 0;
}
