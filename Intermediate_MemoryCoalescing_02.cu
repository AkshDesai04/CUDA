```cpp
/*
Aim of the program:
Write an optimized matrix transpose using shared memory to achieve coalescing.
Each thread block reads a tile from the input matrix into a shared memory buffer
with coalesced reads. Then, after a sync, it writes from shared memory to the
output matrix in a coalesced pattern.

Thinking:
- We use a 2D tile of size TILE_DIM x TILE_DIM.
- Shared memory layout tile[TILE_DIM][TILE_DIM+1] is used to avoid shared‑memory
  bank conflicts (the +1 padding ensures that consecutive rows do not map to
  the same bank when accessed by consecutive threads).
- Threads in a block read the tile from global memory in a coalesced fashion:
  each thread reads one element from a consecutive row of the input matrix.
- After __syncthreads(), threads write the transposed tile back to global
  memory, again with coalesced writes, by swapping the row and column indices.
- The kernel supports square matrices of arbitrary size N. The launch grid
  dimensions are computed as ceil(N/TILE_DIM) for both x and y.
- Host code allocates and initializes a matrix, copies it to the device,
  launches the kernel, copies the result back, and verifies correctness.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define TILE_DIM 32
#define BLOCK_ROWS 32  // same as TILE_DIM for simplicity

// Macro for checking CUDA errors following a kernel launch or API call
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Kernel: optimized matrix transpose using shared memory
__global__ void transposeShared(float *odata, const float *idata, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    // Compute global row and column indices of the element this thread will read
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    // Read from global memory into shared memory (coalesced reads)
    if (xIndex < width && yIndex < height)
    {
        tile[threadIdx.y][threadIdx.x] = idata[yIndex * width + xIndex];
    }

    __syncthreads();

    // Compute transposed indices
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x; // transpose block offset
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write the transposed tile from shared memory to global memory (coalesced writes)
    if (xIndex < height && yIndex < width)
    {
        odata[yIndex * height + xIndex] = tile[threadIdx.x][threadIdx.y];
    }
}

// Utility function to verify correctness
bool verify(const float *h_orig, const float *h_trans, int width, int height)
{
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            float expected = h_orig[i * width + j];
            float actual   = h_trans[j * height + i];
            if (fabs(expected - actual) > 1e-5)
            {
                fprintf(stderr, "Mismatch at (%d,%d): expected %f, got %f\n",
                        i, j, expected, actual);
                return false;
            }
        }
    }
    return true;
}

int main()
{
    const int width  = 1024; // matrix width
    const int height = 1024; // matrix height

    size_t size = width * height * sizeof(float);

    // Allocate host memory
    float *h_idata = (float*)malloc(size);
    float *h_odata = (float*)malloc(size);

    // Initialize input matrix with some values
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            h_idata[i * width + j] = static_cast<float>(i * width + j);
        }
    }

    // Allocate device memory
    float *d_idata = nullptr;
    float *d_odata = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_idata, size));
    CUDA_CHECK(cudaMalloc((void**)&d_odata, size));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockDim(TILE_DIM, BLOCK_ROWS, 1);
    dim3 gridDim((width + TILE_DIM - 1) / TILE_DIM,
                 (height + TILE_DIM - 1) / TILE_DIM,
                 1);

    // Launch kernel
    transposeShared<<<gridDim, blockDim>>>(d_odata, d_idata, width, height);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, size, cudaMemcpyDeviceToHost));

    // Verify result
    if (verify(h_idata, h_odata, width, height))
    {
        printf("Matrix transpose succeeded.\n");
    }
    else
    {
        printf("Matrix transpose failed.\n");
    }

    // Clean up
    free(h_idata);
    free(h_odata);
    CUDA_CHECK(cudaFree(d_idata));
    CUDA_CHECK(cudaFree(d_odata));

    return 0;
}
```