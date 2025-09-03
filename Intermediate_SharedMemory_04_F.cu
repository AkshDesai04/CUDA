```cpp
/*
Profile for different tile sizes (e.g., 8x8, 16x16, 32x32) and find the optimal one for your GPU.

Thinking:
The goal is to benchmark a tiled matrix multiplication kernel in CUDA for a few fixed tile sizes
(8, 16, 32).  To do this we need a single kernel that can work with any tile size, so we use
dynamic shared memory and pass the tile size to the kernel.  The grid and block dimensions
are computed from the tile size, and we allocate the required shared memory per block
(`tileSize * tileSize * 2 * sizeof(float)` for both tiles A and B).

We will:
1. Allocate two input matrices A and B, and one output matrix C on the host.
2. Fill A and B with some deterministic values (e.g., random or a simple pattern) so that
   we can later verify the result if desired.
3. Allocate device memory for A, B, and C.
4. Copy A and B to the device.
5. For each tile size in {8, 16, 32}:
   - Set up blockDim = {tileSize, tileSize} and gridDim accordingly.
   - Record start and stop CUDA events to time the kernel.
   - Launch the kernel with the appropriate shared memory size.
   - Compute elapsed time and print it.
6. Optionally copy the result back and perform a CPU reference calculation for
   correctness checking.
7. Clean up memory.

We use `cudaEventRecord` for timing because it provides high resolution time stamps
and accounts for GPU latency.  Error checking is performed after each CUDA call
so that any failure is reported immediately.

The kernel itself uses the standard tiled matrix multiplication pattern:
   for k in 0 .. N step tileSize
       load a tile of A into shared memory
       load a tile of B into shared memory
       synchronize
       multiply-accumulate
   after the loop, write the result to C (if within bounds).

The code is self-contained and can be compiled with `nvcc -o matmul_tiled matmul_tiled.cu`.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA launch or API call
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Tiled matrix multiplication kernel with dynamic shared memory
__global__ void MatMulTiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int N,
                            int tileSize)
{
    // Compute global row and column indices
    int row = blockIdx.y * tileSize + threadIdx.y;
    int col = blockIdx.x * tileSize + threadIdx.x;

    // Pointers into shared memory for A and B tiles
    extern __shared__ float sharedMem[];
    float* As = sharedMem;                     // size: tileSize * tileSize
    float* Bs = sharedMem + tileSize * tileSize; // size: tileSize * tileSize

    float acc = 0.0f; // Accumulator for the output element

    // Loop over tiles of the matrix
    for (int k = 0; k < N; k += tileSize)
    {
        // Load tile of A into shared memory
        if (row < N && (k + threadIdx.x) < N)
            As[threadIdx.y * tileSize + threadIdx.x] =
                A[row * N + k + threadIdx.x];
        else
            As[threadIdx.y * tileSize + threadIdx.x] = 0.0f;

        // Load tile of B into shared memory
        if (col < N && (k + threadIdx.y) < N)
            Bs[threadIdx.y * tileSize + threadIdx.x] =
                B[(k + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y * tileSize + threadIdx.x] = 0.0f;

        __syncthreads(); // Wait for all loads to complete

        // Multiply the two tiles together
        for (int t = 0; t < tileSize; ++t)
            acc += As[threadIdx.y * tileSize + t] * Bs[t * tileSize + threadIdx.x];

        __syncthreads(); // Wait for all threads to finish before loading next tile
    }

    // Write the result to global memory
    if (row < N && col < N)
        C[row * N + col] = acc;
}

// Simple CPU implementation for verification (optional)
void MatMulCPU(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

int main()
{
    const int N = 1024; // Matrix size NxN
    const int tileSizes[] = {8, 16, 32};
    const int numTileSizes = sizeof(tileSizes) / sizeof(tileSizes[0]);

    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    float* h_C_ref = (float*)malloc(bytes); // For verification

    // Initialize input matrices with random values
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Optional: compute reference result on CPU for verification
    printf("Computing reference result on CPU (this may take a while)...\n");
    MatMulCPU(h_A, h_B, h_C_ref, N);
    printf("CPU computation finished.\n");

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int idx = 0; idx < numTileSizes; ++idx) {
        int tileSize = tileSizes[idx];
        dim3 blockDim(tileSize, tileSize);
        dim3 gridDim((N + tileSize - 1) / tileSize,
                     (N + tileSize - 1) / tileSize);

        size_t sharedMemSize = tileSize * tileSize * 2 * sizeof(float);

        // Record start event
        CUDA_CHECK(cudaEventRecord(start, 0));

        // Launch kernel
        MatMulTiled<<<gridDim, blockDim, sharedMemSize>>>(
            d_A, d_B, d_C, N, tileSize);

        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError());

        // Record stop event
        CUDA_CHECK(cudaEventRecord(stop, 0));

        // Wait for the event to complete
        CUDA_CHECK(cudaEventSynchronize(stop));

        // Calculate elapsed time
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

        // Verify correctness for this tile size
        bool correct = true;
        for (int i = 0; i < N * N && correct; ++i) {
            float diff = fabs(h_C[i] - h_C_ref[i]);
            if (diff > 1e-3f) { // Tolerance
                correct = false;
                printf("Mismatch at index %d: GPU=%f CPU=%f diff=%f\n",
                       i, h_C[i], h_C_ref[i], diff);
            }
        }

        printf("Tile size %3dx%3d: Time = %6.2f ms, %s\n",
               tileSize, tileSize, milliseconds,
               correct ? "Correct" : "Incorrect");
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
}
```