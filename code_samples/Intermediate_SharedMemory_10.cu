```cpp
/*
Aim of the program:
Implement a more efficient shared-memory transpose for a square tile. Thread (x, y) reads from input(x, y) and writes to shared_mem(y, x). This avoids bank conflicts that a simple shared_mem[x][y] access might cause and helps achieve coalesced global memory reads/writes later.

Thinking:
- Use a square tile of size TILE_DIM (e.g., 32). 
- To avoid shared memory bank conflicts when transposing, declare the tile as __shared__ float tile[TILE_DIM][TILE_DIM + 1]; The +1 padding ensures that successive rows are not mapped to the same bank.
- Each thread (tx, ty) in the block loads input[blockIdx.x * TILE_DIM + tx, blockIdx.y * TILE_DIM + ty] into tile[ty][tx].
- Synchronize to ensure all loads are complete.
- Each thread then writes tile[tx][ty] to output[blockIdx.y * TILE_DIM + tx, blockIdx.x * TILE_DIM + ty], which is a coalesced write to the transposed matrix.
- The host code demonstrates allocation, initialization, kernel launch, and verification.

This pattern eliminates bank conflicts for the transposition kernel while keeping global memory accesses coalesced, improving overall performance on CUDA devices.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_DIM 32
#define BLOCK_ROWS TILE_DIM

// Kernel to transpose a matrix using shared memory with padding to avoid bank conflicts
__global__ void transposeShared(float* __restrict__ in, float* __restrict__ out, int width, int height)
{
    // Allocate shared memory tile with padding (+1) to avoid bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load element from global memory into shared memory with transposed indices
    if (x < width && y < height)
    {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }

    __syncthreads();

    // Write transposed element from shared memory to global memory
    int transposed_x = blockIdx.y * TILE_DIM + threadIdx.x;
    int transposed_y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (transposed_x < height && transposed_y < width)
    {
        out[transposed_y * height + transposed_x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Helper function to initialize matrix with random floats
void initMatrix(float* mat, int width, int height)
{
    for (int i = 0; i < width * height; ++i)
    {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Helper function to verify that two matrices are approximately equal
bool verify(const float* a, const float* b, int width, int height)
{
    const float eps = 1e-5f;
    for (int i = 0; i < width * height; ++i)
    {
        if (fabs(a[i] - b[i]) > eps)
        {
            printf("Mismatch at index %d: a=%f, b=%f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    const int width = 1024;   // Number of columns in the input matrix
    const int height = 1024;  // Number of rows in the input matrix

    size_t size = width * height * sizeof(float);

    // Allocate host memory
    float* h_in  = (float*)malloc(size);
    float* h_out = (float*)malloc(size);
    float* h_ref = (float*)malloc(size);

    // Initialize input matrix
    initMatrix(h_in, width, height);

    // Compute reference transpose on CPU for verification
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            h_ref[x * height + y] = h_in[y * width + x];
        }
    }

    // Allocate device memory
    float* d_in;
    float* d_out;
    cudaMalloc((void**)&d_in,  size);
    cudaMalloc((void**)&d_out, size);

    // Copy input to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS);
    dim3 dimGrid((width  + TILE_DIM - 1) / TILE_DIM,
                 (height + TILE_DIM - 1) / TILE_DIM);

    transposeShared<<<dimGrid, dimBlock>>>(d_in, d_out, width, height);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Verify correctness
    if (verify(h_ref, h_out, height, width))
    {
        printf("Transpose successful!\n");
    }
    else
    {
        printf("Transpose failed.\n");
    }

    // Free memory
    free(h_in);
    free(h_out);
    free(h_ref);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
```