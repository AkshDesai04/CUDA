```cpp
/*
Write a naive matrix transpose kernel where thread `(x, y)` reads `input[y][x]` and writes to `output[x][y]`. Assuming row-major layout, analyze why the reads from `input` will be uncoalesced (strided) and hurt performance.

Thinking:
- In row-major storage, consecutive elements in a row are stored next to each other.
- A thread at coordinate (x, y) accesses input[y][x] which means it reads the element from row y, column x.
- If a warp of 32 threads has x varying across 0..31 and a fixed y, they will read input[y][0], input[y][1], ..., input[y][31].
  That is contiguous in memory: good coalescing.
- However, if the thread indexing is such that the warp is distributed across different y values with the same x (e.g., block dim (1,32) or 2D grid with x constant across warp), then each thread reads input[y][x] where y differs.
  These reads become strided: each thread accesses a different row at the same column.
- Strided accesses cause each thread to fetch from separate memory segments, leading to uncoalesced memory transactions.
- This reduces effective memory bandwidth and hurts performance.

The code below implements the naive transpose kernel and demonstrates the uncoalesced access pattern when using a typical 2D thread block layout.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Matrix dimensions
#define N 8  // Use a small matrix for demonstration
#define BLOCK_SIZE 4

__global__ void transpose_naive(const float* __restrict__ in, float* __restrict__ out, int width, int height)
{
    // Thread indices
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index in input
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index in input

    if (x < width && y < height)
    {
        // Read from (y, x) and write to (x, y)
        // Note: row-major layout means element at (row, col) is at in[row * width + col]
        int in_idx  = y * width + x;
        int out_idx = x * height + y; // output is transposed
        out[out_idx] = in[in_idx];
    }
}

int main()
{
    const int width  = N;
    const int height = N;
    const int size   = width * height;
    const size_t bytes = size * sizeof(float);

    // Allocate host memory
    float *h_in  = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);

    // Initialize input matrix
    for (int r = 0; r < height; ++r)
    {
        for (int c = 0; c < width; ++c)
        {
            h_in[r * width + c] = (float)(r * width + c);
        }
    }

    // Allocate device memory
    float *d_in, *d_out;
    cudaMalloc(&d_in,  bytes);
    cudaMalloc(&d_out, bytes);

    // Copy input to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width  + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    transpose_naive<<<gridDim, blockDim>>>(d_in, d_out, width, height);

    // Copy result back to host
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Print input and transposed output
    printf("Input matrix (%dx%d):\n", height, width);
    for (int r = 0; r < height; ++r)
    {
        for (int c = 0; c < width; ++c)
        {
            printf("%6.1f ", h_in[r * width + c]);
        }
        printf("\n");
    }

    printf("\nTransposed matrix (%dx%d):\n", width, height);
    for (int r = 0; r < width; ++r)
    {
        for (int c = 0; c < height; ++c)
        {
            printf("%6.1f ", h_out[r * height + c]);
        }
        printf("\n");
    }

    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
```