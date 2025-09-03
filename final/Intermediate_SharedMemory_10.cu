```cuda
/*
Aim of the program:
Implement a more efficient shared-memory transpose for a square tile.
Thread `(x, y)` reads from `input(x, y)` and writes to `shared_mem(y, x)`. 
This avoids bank conflicts that a simple `shared_mem[x][y]` access might cause
and helps achieve coalesced global memory reads/writes later.

Thinking:
- The core idea is to load a tile of the input matrix into shared memory
  using transposed indices, i.e. `shared[threadIdx.y][threadIdx.x]` is set
  to the value at global coordinates `(x, y)`. This pattern ensures that
  when threads read from shared memory they hit different banks,
  reducing bank conflicts.
- After loading, a synchronization barrier (`__syncthreads()`) guarantees
  that all data is available before any thread starts writing the
  transposed tile to global memory.
- The write phase uses the opposite transposition: each thread writes
  the element it previously stored in shared memory but now from the
  transposed indices. This guarantees coalesced writes because the
  destination indices follow a regular stride pattern.
- We add one extra column to the shared-memory tile (`TILE_DIM+1`) to
  further eliminate bank conflicts. This is a common trick for 32-bit
  elements on GPUs with 32 banks.
- For simplicity, the program operates on a single square matrix of
  floating‑point values. The kernel is launched with a grid of
  `TILE_DIM`×`TILE_DIM` blocks so that each block handles a tile.
- The host code allocates a small matrix (e.g. 1024×1024), initializes
  it, copies it to the device, runs the transpose kernel, copies the
  result back, and optionally prints a small section to verify
  correctness.

The final file is a self‑contained CUDA source (`.cu`) that can be
compiled with `nvcc` and executed on any CUDA‑capable GPU.
*/

#include <iostream>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a kernel launch or API call
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__      \
                      << " : " << cudaGetErrorString(err) << std::endl;       \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

constexpr int TILE_DIM = 32;   // Size of the tile (square)

__global__ void transpose(float *out, const float *in, int width, int height)
{
    // Shared memory tile with an extra column to avoid bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Global indices of the element this thread will read
    int x_in = blockIdx.x * TILE_DIM + threadIdx.x;
    int y_in = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load into shared memory with transposed indices
    if (x_in < width && y_in < height) {
        tile[threadIdx.y][threadIdx.x] = in[y_in * width + x_in];
    }

    __syncthreads(); // Ensure all reads are complete before writes

    // Global indices of the transposed location to write to
    int x_out = blockIdx.y * TILE_DIM + threadIdx.x;
    int y_out = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write the transposed tile from shared memory
    if (x_out < height && y_out < width) {
        out[y_out * height + x_out] = tile[threadIdx.x][threadIdx.y];
    }
}

int main()
{
    const int N = 1024; // Matrix dimension (N x N)
    const int size = N * N;
    const size_t bytes = size * sizeof(float);

    // Host allocation
    float *h_in  = new float[size];
    float *h_out = new float[size];

    // Initialize input matrix with simple pattern for verification
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            h_in[y * N + x] = static_cast<float>(y * N + x);
        }
    }

    // Device allocation
    float *d_in = nullptr;
    float *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize((N + TILE_DIM - 1) / TILE_DIM,
                  (N + TILE_DIM - 1) / TILE_DIM);
    transpose<<<gridSize, blockSize>>>(d_out, d_in, N, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Simple verification: print a few elements
    std::cout << "Sample values (input -> output):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        int x = i, y = i;
        int idx_in = y * N + x;
        int idx_out = x * N + y;
        std::cout << "h_in[" << y << "][" << x << "] = " << h_in[idx_in]
                  << " -> h_out[" << x << "][" << y << "] = " << h_out[idx_out] << std::endl;
    }

    // Clean up
    delete[] h_in;
    delete[] h_out;
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
```