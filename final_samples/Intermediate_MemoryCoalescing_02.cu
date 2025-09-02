/*
Write an optimized matrix transpose using shared memory to achieve coalescing. Each thread block reads a tile from the input matrix into a shared memory buffer with coalesced reads. Then, after a sync, it writes from shared memory to the output matrix in a coalesced pattern.

Thinking:
- Use a 2‑D CUDA kernel with a square tile size that matches the warp size (32) to simplify indexing and to keep all threads active.
- The shared memory tile is declared as tile[TILE_DIM][TILE_DIM+1] – the extra column is to avoid bank conflicts when accessing tile[threadIdx.x][threadIdx.y].
- Coalesced reads: for each thread in a block, the global address accessed in the input matrix is
      idx = (blockIdx.y * TILE_DIM + threadIdx.y) * width + (blockIdx.x * TILE_DIM + threadIdx.x)
  Within a row of the tile, threadIdx.x increments by 1, giving consecutive memory accesses – which is coalesced.
- After loading the tile, we synchronize.
- Coalesced writes: we write the transposed element by swapping the blockIdx.x and blockIdx.y indices and by using the same ordering of threadIdx.x/threadIdx.y. The output index becomes:
      idx_out = (blockIdx.x * TILE_DIM + threadIdx.y) * height + (blockIdx.y * TILE_DIM + threadIdx.x)
  Again, within a row of the transposed tile, threadIdx.x increments by 1, giving consecutive writes – also coalesced.
- Boundary handling: check if the computed global indices are inside the matrix bounds before accessing memory.
- Host side: allocate host and device buffers, copy data, launch the kernel with a grid that covers the whole matrix, and copy back the result.
- Simple verification: compare a few elements of the transposed matrix with the expected values.
*/

#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_DIM 32

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",      \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),           \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

// Kernel: transpose matrix using shared memory
__global__ void transpose_kernel(float *o, const float *i, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x; // column in input
    int y = blockIdx.y * TILE_DIM + threadIdx.y; // row in input
    int idx_in = y * width + x;

    // Read from global memory to shared memory (coalesced)
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = i[idx_in];
    }
    __syncthreads();

    // Write transposed tile back to global memory (coalesced)
    x = blockIdx.y * TILE_DIM + threadIdx.x; // column in output
    y = blockIdx.x * TILE_DIM + threadIdx.y; // row in output
    int idx_out = y * height + x; // Note: output dimensions swapped

    if (x < height && y < width) {
        o[idx_out] = tile[threadIdx.x][threadIdx.y];
    }
}

// Simple function to print matrix for debugging
void print_matrix(const float *mat, int width, int height)
{
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            printf("%5.1f ", mat[r * width + c]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void)
{
    const int width = 1024;   // input matrix width
    const int height = 512;   // input matrix height

    size_t input_size = width * height * sizeof(float);
    size_t output_size = width * height * sizeof(float);

    // Allocate host memory
    float *h_input = (float *)malloc(input_size);
    float *h_output = (float *)malloc(output_size);

    // Initialize input matrix with a pattern
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            h_input[r * width + c] = (float)(r * width + c);
        }
    }

    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void **)&d_input, input_size));
    CUDA_CHECK(cudaMalloc((void **)&d_output, output_size));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((width + TILE_DIM - 1) / TILE_DIM,
                 (height + TILE_DIM - 1) / TILE_DIM);

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch kernel
    transpose_kernel<<<gridDim, blockDim>>>(d_output, d_input, width, height);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));

    // Simple verification: print few elements
    printf("Transposed matrix (sample):\n");
    for (int r = 0; r < 5; ++r) {
        for (int c = 0; c < 5; ++c) {
            int idx = r * height + c; // because output dimensions swapped
            printf("%5.1f ", h_output[idx]);
        }
        printf("\n");
    }

    printf("\nKernel execution time: %.3f ms\n", milliseconds);

    // Clean up
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
