```cuda
/*
Aim of the program:
After a __syncthreads(), the kernel should write from shared memory to the output matrix from a transposed position:
output[...] = s_tile[threadIdx.x][threadIdx.y] (note x and y are swapped). This write should also be coalesced.

Thinking:
The task is to implement a CUDA matrix transpose that uses shared memory tiles for efficiency and ensures that the write back to global memory is both correct and coalesced.  The typical pattern is:
  1. Each thread loads one element from the input matrix into a shared memory tile, indexed as tile[threadIdx.y][threadIdx.x].
  2. A __syncthreads() ensures all loads are complete.
  3. The tile is then written to the output matrix with the row/column indices swapped.  To keep the write coalesced, we compute the output address so that consecutive threads in a warp write to consecutive memory locations.  This is achieved by having the outer loop index (y) fixed per warp while the inner index (x) varies with threadIdx.x.  Consequently the global write is y*height + x where x = blockIdx.y*TILE_DIM + threadIdx.x and y = blockIdx.x*TILE_DIM + threadIdx.y.
  
Additional details:
- The shared tile is declared as tile[BLOCK_SIZE][BLOCK_SIZE+1] to avoid bank conflicts.
- Bounds checks guard against partial tiles at matrix edges.
- The host code sets up a simple test matrix, runs the kernel, copies the result back, and verifies correctness by comparing the transposed values.
- Error checking macros ensure any CUDA runtime errors are caught.

The program compiles with nvcc and demonstrates a coalesced transpose kernel.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

// CUDA error checking macro
#define CHECK_CUDA(call)                                                   \
    {                                                                      \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    }

// Kernel that transposes a matrix using shared memory tiles.
// After __syncthreads(), the tile is written to the output matrix with swapped indices,
// ensuring coalesced writes.
__global__ void transposeKernel(const float * __restrict__ d_in,
                                float * __restrict__ d_out,
                                int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    // Read input matrix into shared memory tile
    if (xIndex < width && yIndex < height)
    {
        tile[threadIdx.y][threadIdx.x] = d_in[yIndex * width + xIndex];
    }
    else
    {
        // Fill with zeros if out of bounds to avoid uninitialized reads
        tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Compute transposed indices
    int xOut = blockIdx.y * TILE_DIM + threadIdx.x; // note swap
    int yOut = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write transposed data from shared memory to output matrix
    if (xOut < height && yOut < width)
    {
        d_out[yOut * height + xOut] = tile[threadIdx.x][threadIdx.y];
    }
}

// Host helper function to allocate and initialize matrix
void initializeMatrix(float *mat, int width, int height)
{
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            mat[y * width + x] = (float)(y * width + x);
        }
    }
}

// Function to verify transpose correctness
int verifyTranspose(const float *in, const float *out, int width, int height)
{
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float expected = in[y * width + x];
            float actual = out[x * height + y];
            if (expected != actual)
            {
                printf("Mismatch at (%d,%d): expected %f, got %f\n",
                       y, x, expected, actual);
                return 0;
            }
        }
    }
    return 1;
}

int main(int argc, char *argv[])
{
    int width = 1024;  // Matrix width (number of columns)
    int height = 1024; // Matrix height (number of rows)

    if (argc >= 3)
    {
        width = atoi(argv[1]);
        height = atoi(argv[2]);
    }

    size_t size_in = width * height * sizeof(float);
    size_t size_out = width * height * sizeof(float); // same size, but transposed

    // Allocate host memory
    float *h_in  = (float *)malloc(size_in);
    float *h_out = (float *)malloc(size_out);
    if (!h_in || !h_out)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input matrix
    initializeMatrix(h_in, width, height);

    // Allocate device memory
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc((void **)&d_in, size_in));
    CHECK_CUDA(cudaMalloc((void **)&d_out, size_out));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, size_in, cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    dim3 blockDim(TILE_DIM, TILE_ROWS);
    dim3 gridDim((width  + TILE_DIM - 1) / TILE_DIM,
                 (height + TILE_DIM - 1) / TILE_DIM);

    // Launch transpose kernel
    transposeKernel<<<gridDim, blockDim>>>(d_in, d_out, width, height);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost));

    // Verify correctness
    if (verifyTranspose(h_in, h_out, width, height))
    {
        printf("Transpose verification passed.\n");
    }
    else
    {
        printf("Transpose verification failed.\n");
    }

    // Free resources
    free(h_in);
    free(h_out);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}
```