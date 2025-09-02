```cuda
/* 
Aim of the program: 
Implement a more efficient shared-memory transpose for a square tile. Thread (x, y) reads from input(x, y) and writes to shared_mem(y, x). This avoids bank conflicts that a simple shared_mem[x][y] access might cause and helps achieve coalesced global memory reads/writes later.

Thinking:
- We want to transpose a square matrix of size N x N using a tiled approach.
- Each tile is of size TILE_SIZE x TILE_SIZE, e.g. 32 for full warp usage.
- To avoid shared‑memory bank conflicts we pad each row with an extra element (TILE_SIZE+1 columns). This ensures that accesses tile[threadIdx.y][threadIdx.x] and tile[threadIdx.x][threadIdx.y] map to different banks.
- Kernel reads from global memory in a coalesced manner: each thread reads one element per warp.
- After synchronizing, each thread writes the transposed value to global memory. The write pattern is also coalesced because the threads in a block now traverse the transposed tile row‑major.
- Host code allocates input, fills it, copies to device, launches the kernel with appropriate grid/block dimensions, copies result back and verifies correctness.

The code below demonstrates this approach for a float matrix.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define TILE_SIZE 32

// Kernel: transpose using shared memory with padding to avoid bank conflicts
__global__ void transpose_shared(const float *in, float *out, int width)
{
    // Shared memory tile with padding: TILE_SIZE rows, TILE_SIZE+1 columns
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    // Calculate global coordinates of the element this thread will load
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Load element into shared memory with transposed indices
    if (x < width && y < width) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }

    __syncthreads();

    // Write the transposed element from shared memory to global memory
    // Note: now blockIdx.x and blockIdx.y are swapped to produce correct output ordering
    x = blockIdx.y * TILE_SIZE + threadIdx.x; // new x coordinate after transpose
    y = blockIdx.x * TILE_SIZE + threadIdx.y; // new y coordinate after transpose

    if (x < width && y < width) {
        out[y * width + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Utility function for checking CUDA errors
inline void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    const int N = 512; // matrix dimension (must be multiple of TILE_SIZE for simplicity)
    const size_t size = N * N * sizeof(float);

    // Allocate host memory
    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);
    if (!h_in || !h_out) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize input matrix with some pattern
    for (int i = 0; i < N * N; ++i) {
        h_in[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_in = nullptr;
    float *d_out = nullptr;
    checkCuda(cudaMalloc(&d_in, size), "cudaMalloc d_in");
    checkCuda(cudaMalloc(&d_out, size), "cudaMalloc d_out");

    // Copy input to device
    checkCuda(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    // Define block and grid dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    transpose_shared<<<gridDim, blockDim>>>(d_in, d_out, N);
    checkCuda(cudaGetLastError(), "Kernel launch");

    // Copy result back to host
    checkCuda(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    // Verify correctness
    bool ok = true;
    for (int y = 0; y < N && ok; ++y) {
        for (int x = 0; x < N; ++x) {
            if (fabs(h_out[y * N + x] - h_in[x * N + y]) > 1e-5f) {
                ok = false;
                fprintf(stderr, "Mismatch at (%d,%d): got %f, expected %f\n",
                        y, x, h_out[y * N + x], h_in[x * N + y]);
                break;
            }
        }
    }

    if (ok) {
        printf("Transpose successful, results match.\n");
    } else {
        printf("Transpose failed.\n");
    }

    // Clean up
    free(h_in);
    free(h_out);
    checkCuda(cudaFree(d_in), "cudaFree d_in");
    checkCuda(cudaFree(d_out), "cudaFree d_out");

    return 0;
}
```