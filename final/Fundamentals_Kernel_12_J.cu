/*
Aim of the program: 
Explain in a comment why calculating the number of blocks as (N + BS - 1) / BS is better than N / BS.

Thinking:
When launching a CUDA kernel we need to cover all N elements.  If we simply compute the number of blocks with N / BS (integer division), we lose any elements that don't fill a complete block: the remainder elements are left uncovered.  This would require us to add an explicit check inside the kernel to skip out‑of‑range threads, but even then we would still launch fewer blocks than needed, potentially causing race conditions or missing work.  The expression (N + BS - 1) / BS is a common idiom that rounds up the division: it adds BS‑1 to N before dividing, ensuring that any non‑zero remainder forces an additional block.  Thus every element gets a thread and we avoid the need for conditional bounds checks for the block count itself.  It is more robust and simpler, especially when the kernel performs a bounds check anyway for safety.  In short, (N + BS - 1) / BS guarantees full coverage of the data set without under‑allocation.
*/

#include <stdio.h>
#include <cuda.h>

__global__ void dummyKernel(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = idx;
    }
}

int main() {
    const int N = 1000;
    const int BS = 256;
    int *d_data;
    size_t size = N * sizeof(int);
    cudaMalloc(&d_data, size);

    // Calculate number of blocks using the robust formula
    int numBlocks = (N + BS - 1) / BS;

    printf("Launching kernel with %d blocks (BS=%d) to cover %d elements.\n", numBlocks, BS, N);

    dummyKernel<<<numBlocks, BS>>>(d_data, N);
    cudaDeviceSynchronize();

    // Clean up
    cudaFree(d_data);
    return 0;
}
