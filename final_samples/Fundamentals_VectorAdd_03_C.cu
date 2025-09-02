/*
Explain in a comment why the grid calculation `(N + BLOCK_SIZE - 1) / BLOCK_SIZE` launches enough blocks to cover all elements.

In CUDA, each block contains a fixed number of threads, defined here as BLOCK_SIZE. To process N elements, we need enough blocks so that the total number of threads (blocks × BLOCK_SIZE) is at least N. The expression `(N + BLOCK_SIZE - 1) / BLOCK_SIZE` performs integer division that rounds up to the next whole block when N is not an exact multiple of BLOCK_SIZE. If N is divisible by BLOCK_SIZE, the addition of `BLOCK_SIZE - 1` does not change the result: `(k*BLOCK_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE = k`. If there is a remainder r (0 < r < BLOCK_SIZE), adding BLOCK_SIZE - 1 ensures the numerator becomes at least `k*BLOCK_SIZE + BLOCK_SIZE`, which when divided by BLOCK_SIZE yields `k+1`. Thus this formula guarantees that the number of blocks launched is the smallest integer greater than or equal to N/BLOCK_SIZE, ensuring that every element has at least one thread assigned, and no elements are left unprocessed. The kernel itself checks `if (idx < N)` to avoid out-of-bounds accesses for the last block, which may contain unused threads if N is not an exact multiple of BLOCK_SIZE.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void add_kernel(const int *a, const int *b, int *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void)
{
    int N = 1000; // Number of elements
    size_t size = N * sizeof(int);

    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);

    for (int i = 0; i < N; ++i)
    {
        h_a[i] = i;
        h_b[i] = N - i;
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_kernel<<<grid, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < 10; ++i)
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
