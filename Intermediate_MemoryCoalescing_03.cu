```cpp
/*
Aim of the program (as given in the prompt):
"Profile the naive and shared-memory-based transpose kernels. Use nvprof or NVIDIA Nsight Compute to measure the global memory transaction count and observed bandwidth for both. The difference should be significant."

Thinking process:

1. I need to implement two CUDA kernels that perform matrix transposition.
   - The first kernel is a naive implementation where each thread reads an element from global memory and writes it to the transposed location. This will cause many uncoalesced memory transactions, leading to lower bandwidth.
   - The second kernel uses shared memory tiles to coalesce reads/writes. By loading a TILE_DIM x TILE_DIM block into shared memory, performing a transpose in shared memory, and then writing back, we should see a significant improvement in global memory transaction count and observed bandwidth.

2. To compare them, the host program will:
   - Allocate a square matrix of size N x N (e.g., 1024 x 1024) on the host and initialize it with known values.
   - Allocate device memory for the input and output matrices.
   - Launch the naive kernel, copy the result back, and verify correctness.
   - Launch the shared-memory kernel, copy the result back, and verify correctness.
   - Measure execution time for each kernel using CUDA events for additional context (though the main metric is measured by nvprof/Nsight).

3. For the shared-memory kernel:
   - Use TILE_DIM = 32 and a shared memory array of size TILE_DIM x (TILE_DIM + 1). The extra column avoids bank conflicts.
   - Each thread block processes a TILE_DIM x TILE_DIM tile. Threads load data into shared memory and then write transposed data back to global memory.

4. Boundary conditions:
   - The code will assume that N is divisible by TILE_DIM to keep the implementation simple. For a production version, one would handle the remaining elements with conditionals.

5. The program prints basic information (matrix size, memory usage, execution time) so the user can confirm that the kernels ran and that the shared-memory version is faster.

6. The profiling of global memory transaction count and observed bandwidth will be done externally using nvprof or Nsight Compute, as requested.

The following code implements all of the above. Compile with:
    nvcc -O2 -arch=sm_70 transpose.cu -o transpose
Run with:
    ./transpose
and then profile with:
    nvprof --print-gpu-trace ./transpose
or Nsight Compute.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 1024                 // Matrix dimension (must be divisible by TILE_DIM)
#define TILE_DIM 32            // Tile size for shared-memory transpose
#define BLOCK_ROWS TILE_DIM    // Number of rows processed per block

/* ------------------------------------------------------------------------
   Naive transpose kernel: each thread reads an element and writes to
   its transposed position directly in global memory.
   ------------------------------------------------------------------------ */
__global__ void transpose_naive(const float *in, float *out, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (x < width && y < height)
    {
        int in_idx  = y * width + x;
        int out_idx = x * height + y; // Transposed position
        out[out_idx] = in[in_idx];
    }
}

/* ------------------------------------------------------------------------
   Shared-memory transpose kernel: uses a TILE_DIM x TILE_DIM tile loaded
   into shared memory. The extra column in shared memory (TILE_DIM+1)
   prevents shared memory bank conflicts.
   ------------------------------------------------------------------------ */
__global__ void transpose_shared(const float *in, float *out, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load data into shared memory
    if (x < width && y < height)
    {
        int in_idx = y * width + x;
        tile[threadIdx.y][threadIdx.x] = in[in_idx];
    }

    __syncthreads();

    // Write transposed data from shared memory to global memory
    x = blockIdx.y * TILE_DIM + threadIdx.x; // Transposed coordinates
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < height && y < width) // Note: width and height may differ
    {
        int out_idx = y * height + x;
        out[out_idx] = tile[threadIdx.x][threadIdx.y];
    }
}

/* ------------------------------------------------------------------------
   Helper function to check CUDA errors.
   ------------------------------------------------------------------------ */
void checkCudaError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* ------------------------------------------------------------------------
   Host function to run the transpose kernels and measure execution time.
   ------------------------------------------------------------------------ */
void runTranspose()
{
    const int width = N;
    const int height = N;
    const size_t matrixSize = width * height * sizeof(float);

    // Allocate host matrices
    float *h_in  = (float*)malloc(matrixSize);
    float *h_out_naive = (float*)malloc(matrixSize);
    float *h_out_shared = (float*)malloc(matrixSize);
    if (!h_in || !h_out_naive || !h_out_shared)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize input matrix with known values
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            h_in[y * width + x] = (float)(y * width + x);

    // Allocate device matrices
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, matrixSize);
    cudaMalloc((void**)&d_out, matrixSize);
    checkCudaError("cudaMalloc");

    // Copy input to device
    cudaMemcpy(d_in, h_in, matrixSize, cudaMemcpyHostToDevice);
    checkCudaError("cudaMemcpy H2D");

    // Define grid and block dimensions
    dim3 blockDim(TILE_DIM, BLOCK_ROWS);
    dim3 gridDim((width  + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ==================== Naive kernel ====================
    cudaEventRecord(start, 0);
    transpose_naive<<<gridDim, blockDim>>>(d_in, d_out, width, height);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    checkCudaError("Naive kernel launch");

    float ms_naive = 0;
    cudaEventElapsedTime(&ms_naive, start, stop);

    // Copy result back to host
    cudaMemcpy(h_out_naive, d_out, matrixSize, cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy D2H (naive)");

    // ==================== Shared-memory kernel ====================
    cudaEventRecord(start, 0);
    transpose_shared<<<gridDim, blockDim>>>(d_in, d_out, width, height);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    checkCudaError("Shared kernel launch");

    float ms_shared = 0;
    cudaEventElapsedTime(&ms_shared, start, stop);

    // Copy result back to host
    cudaMemcpy(h_out_shared, d_out, matrixSize, cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy D2H (shared)");

    // ==================== Verification ====================
    int mismatches = 0;
    for (int i = 0; i < width * height; ++i)
    {
        if (h_out_naive[i] != h_in[i])
            ++mismatches;
        if (h_out_shared[i] != h_in[i])
            ++mismatches;
    }
    if (mismatches == 0)
        printf("Verification passed: both kernels produced correct transposition.\n");
    else
        printf("Verification failed: %d mismatches found.\n", mismatches);

    // ==================== Summary ====================
    printf("\n=== Transpose Performance Summary ===\n");
    printf("Matrix size: %d x %d\n", width, height);
    printf("Memory per matrix: %.2f MB\n", matrixSize / (1024.0 * 1024.0));
    printf("Naive kernel time: %.3f ms\n", ms_naive);
    printf("Shared-memory kernel time: %.3f ms\n", ms_shared);
    printf("Speedup: %.2fx\n", ms_naive / ms_shared);

    // ==================== Cleanup ====================
    free(h_in);
    free(h_out_naive);
    free(h_out_shared);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/* ------------------------------------------------------------------------
   Main function
   ------------------------------------------------------------------------ */
int main()
{
    printf("Starting transpose kernels on %d x %d matrix.\n", N, N);
    runTranspose();
    printf("Done.\n");
    return 0;
}
```