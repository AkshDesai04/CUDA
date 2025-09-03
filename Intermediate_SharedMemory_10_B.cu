/*
After a __syncthreads(), the kernel should write from shared memory to the output matrix from a transposed position: output[...] = s_tile[threadIdx.x][threadIdx.y] (note x and y are swapped). This write should also be coalesced.

Thought process:

- To achieve a coalesced write while transposing, we arrange that threads in a warp vary threadIdx.x while threadIdx.y stays constant. In row‑major memory, varying the column index (i.e. the second coordinate) by 1 gives consecutive memory addresses. Therefore, we will write to the output using the formula: 
  output[(blockIdx.y * TILE_DIM + threadIdx.x) * height + (blockIdx.x * TILE_DIM + threadIdx.y)] = s_tile[threadIdx.x][threadIdx.y];
  Here `height` is the original input width, so the output matrix dimensions are swapped.

- We load the tile into shared memory in a transposed fashion: s_tile[threadIdx.x][threadIdx.y] = input[y * width + x], where x and y are the global coordinates of the element being read. This stores the element at (x, y) of the input into shared memory at position (x, y), so that after the sync, we can directly write it to the transposed location in the output.

- The kernel includes bounds checking to handle cases where the matrix dimensions are not multiples of TILE_DIM.

- A simple host routine demonstrates allocation, initialization, kernel launch, and verification of the transpose.

*/

#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

// Error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Kernel that performs a matrix transpose using shared memory with swapped indices.
__global__ void transposeKernel(const float * __restrict__ d_in, float * __restrict__ d_out,
                                int width, int height)
{
    // Shared memory tile with padding to avoid bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Global indices of the element to read
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load from global memory to shared memory in transposed manner
    if (x < width && y < height)
    {
        tile[threadIdx.x][threadIdx.y] = d_in[y * width + x];
    }
    else
    {
        // For out‑of‑bounds threads, set to 0 (or any placeholder)
        tile[threadIdx.x][threadIdx.y] = 0.0f;
    }

    __syncthreads();

    // Compute destination indices in the transposed matrix
    int dstX = blockIdx.y * TILE_DIM + threadIdx.x; // row in output
    int dstY = blockIdx.x * TILE_DIM + threadIdx.y; // column in output

    // Write back to global memory (coalesced writes)
    if (dstX < height && dstY < width)
    {
        d_out[dstX * height + dstY] = tile[threadIdx.x][threadIdx.y];
    }
}

// Simple host routine to test the transpose kernel
int main(void)
{
    const int width  = 256;  // input matrix width
    const int height = 256;  // input matrix height

    size_t bytes_in  = width  * height * sizeof(float);
    size_t bytes_out = width  * height * sizeof(float); // transposed matrix has same total size

    // Allocate host memory
    float *h_in  = (float*)malloc(bytes_in);
    float *h_out = (float*)malloc(bytes_out);

    // Initialize input with some pattern
    for (int i = 0; i < width * height; ++i)
    {
        h_in[i] = (float)i;
    }

    // Allocate device memory
    float *d_in  = NULL;
    float *d_out = NULL;
    cudaCheckError(cudaMalloc((void**)&d_in,  bytes_in));
    cudaCheckError(cudaMalloc((void**)&d_out, bytes_out));

    // Copy input to device
    cudaCheckError(cudaMemcpy(d_in, h_in, bytes_in, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid((width + TILE_DIM - 1) / TILE_DIM,
                 (height + TILE_DIM - 1) / TILE_DIM);

    transposeKernel<<<dimGrid, dimBlock>>>(d_in, d_out, width, height);
    cudaCheckError(cudaGetLastError());

    // Copy result back to host
    cudaCheckError(cudaMemcpy(h_out, d_out, bytes_out, cudaMemcpyDeviceToHost));

    // Verify a few elements
    bool ok = true;
    for (int i = 0; i < 10; ++i)
    {
        int row = i;
        int col = i;
        int idx_in  = row * width + col;
        int idx_out = col * height + row; // transposed
        if (h_out[idx_out] != h_in[idx_in])
        {
            printf("Mismatch at (%d,%d): in=%f out=%f\n",
                   row, col, h_in[idx_in], h_out[idx_out]);
            ok = false;
            break;
        }
    }

    if (ok)
    {
        printf("Transpose successful for first 10 elements.\n");
    }

    // Free memory
    free(h_in);
    free(h_out);
    cudaCheckError(cudaFree(d_in));
    cudaCheckError(cudaFree(d_out));

    return 0;
}
