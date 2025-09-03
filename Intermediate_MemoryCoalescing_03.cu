```cuda
/*
Aim: Profile the naive and shared-memory-based transpose kernels.
The program launches both kernels on a square matrix of floats,
measures execution time, and calculates the observed memory bandwidth.
Use nvprof or NVIDIA Nsight Compute to inspect global memory transaction
count and bandwidth for each kernel. The shared-memory kernel should show
a significant reduction in global memory transactions and higher
bandwidth.

Thinking:
- Use a square matrix of size N x N.
- Allocate host matrix, fill with random data.
- Allocate device buffers using cudaMallocPitch if needed; for simplicity use contiguous.
- Implement naive transpose: each thread copies one element.
- Implement shared-memory transpose: load tile into shared memory with padding to avoid bank conflicts,
  then write transposed tile.
- Use block size TILE_DIM x BLOCK_ROWS.
- Measure execution time with cudaEvent.
- Compute bytes transferred: input + output = 2*N*N*sizeof(float).
- Bandwidth = bytes / time.
- Verify correctness by comparing host and device results.
- Output timings and bandwidth.
- Compile with nvcc and run; use nvprof or Nsight Compute for detailed profiling.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                     \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s (%d): %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

const int N = 4096;          // Matrix dimension (NxN)
const int TILE_DIM = 32;     // Tile width for shared memory transpose
const int BLOCK_ROWS = 8;    // Number of rows per block in shared memory kernel

// Naive transpose kernel: each thread reads one element from in and writes to out
__global__ void transposeNaive(const float *in, float *out, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // Column index in input
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // Row index in input

    if (x < width && y < height) {
        int index_in  = y * width + x;
        int index_out = x * height + y;  // Transposed position
        out[index_out] = in[index_in];
    }
}

// Shared-memory transpose kernel with padding to avoid bank conflicts
__global__ void transposeShared(float *out, const float *in, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = yIndex * width + xIndex;

    // Load data from global memory into shared memory
    if (xIndex < width && yIndex < height) {
        tile[threadIdx.y][threadIdx.x] = in[index_in];
    }

    __syncthreads();

    // Write transposed data from shared memory to global memory
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = yIndex * height + xIndex;  // Transposed position

    if (xIndex < height && yIndex < width) {
        out[index_out] = tile[threadIdx.x][threadIdx.y];
    }
}

// Helper to check if two matrices are equal
int verifyResult(const float *h_gold, const float *h_test, int size)
{
    const float epsilon = 1e-5f;
    for (int i = 0; i < size; ++i) {
        float diff = fabsf(h_gold[i] - h_test[i]);
        if (diff > epsilon) {
            printf("Mismatch at index %d: gold=%f, test=%f\n", i, h_gold[i], h_test[i]);
            return 0;
        }
    }
    return 1;
}

int main(void)
{
    size_t bytes = N * N * sizeof(float);
    float *h_in  = (float*)malloc(bytes);
    float *h_out_naive  = (float*)malloc(bytes);
    float *h_out_shared = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);  // Reference result

    // Initialize input matrix with random data
    for (int i = 0; i < N * N; ++i) {
        h_in[i] = rand() / (float)RAND_MAX;
    }

    // Reference transpose on host
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            h_ref[col * N + row] = h_in[row * N + col];
        }
    }

    // Device memory allocation
    float *d_in, *d_out;
    CHECK(cudaMalloc((void**)&d_in, bytes));
    CHECK(cudaMalloc((void**)&d_out, bytes));

    // Copy input to device
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    // Timing events
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // 1) Naive kernel
    CHECK(cudaEventRecord(start));
    transposeNaive<<<gridDim, blockDim>>>(d_in, d_out, N, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms_naive = 0.0f;
    CHECK(cudaEventElapsedTime(&ms_naive, start, stop));

    CHECK(cudaMemcpy(h_out_naive, d_out, bytes, cudaMemcpyDeviceToHost));

    // 2) Shared-memory kernel
    CHECK(cudaEventRecord(start));
    transposeShared<<<gridDim, blockDim>>>(d_out, d_in, N, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms_shared = 0.0f;
    CHECK(cudaEventElapsedTime(&ms_shared, start, stop));

    CHECK(cudaMemcpy(h_out_shared, d_out, bytes, cudaMemcpyDeviceToHost));

    // Verification
    if (!verifyResult(h_ref, h_out_naive, N * N)) {
        fprintf(stderr, "Naive kernel result mismatch!\n");
    } else {
        printf("Naive kernel result verified.\n");
    }

    if (!verifyResult(h_ref, h_out_shared, N * N)) {
        fprintf(stderr, "Shared-memory kernel result mismatch!\n");
    } else {
        printf("Shared-memory kernel result verified.\n");
    }

    // Compute bandwidth (bytes transferred / time)
    double time_sec_naive  = ms_naive  * 1e-3;
    double time_sec_shared = ms_shared * 1e-3;
    double bytes_transferred = 2.0 * bytes;  // input + output

    double bw_naive  = bytes_transferred / time_sec_naive  / 1e9; // GB/s
    double bw_shared = bytes_transferred / time_sec_shared / 1e9; // GB/s

    printf("\n--- Performance Summary ---\n");
    printf("Naive transpose:\n");
    printf("  Time: %.3f ms\n", ms_naive);
    printf("  Bandwidth: %.3f GB/s\n", bw_naive);
    printf("Shared-memory transpose:\n");
    printf("  Time: %.3f ms\n", ms_shared);
    printf("  Bandwidth: %.3f GB/s\n", bw_shared);

    // Clean up
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out_naive);
    free(h_out_shared);
    free(h_ref);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return 0;
}
```