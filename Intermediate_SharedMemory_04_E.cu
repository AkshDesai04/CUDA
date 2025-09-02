```cuda
/********************************************************************
 * Aim of the program: Analyze the instruction mix reported by the
 * profiler. The tiled version should have a higher ratio of arithmetic
 * instructions to memory instructions.
 *
 * Thought process:
 * 1. The task is to produce a CUDA program that allows a user to
 *    compare instruction mixes between two versions of a kernel.  
 * 2. The natural candidate is matrix multiplication because it
 *    involves a lot of arithmetic (multiply‑add) and memory
 *    accesses.  A naive implementation accesses global memory
 *    repeatedly, whereas a tiled (shared‑memory) version reuses
 *    data, reducing memory traffic and increasing arithmetic
 *    density.
 * 3. The program will therefore implement:
 *      - A naive matrix multiplication kernel `matMulNaive`.
 *      - A tiled matrix multiplication kernel `matMulTiled` that
 *        uses shared memory and loop tiling.
 * 4. The host code will:
 *      - Allocate two input matrices A and B and one output matrix C.
 *      - Initialize A and B with random values.
 *      - Call each kernel, measuring execution time with CUDA events.
 *      - Copy the result back to host and optionally verify correctness.
 * 5. The user can run `nvprof` or Nsight Systems/Compute to view the
 *    instruction mix (e.g., % of arithmetic vs. memory instructions)
 *    for each kernel.  The tiled kernel should show a higher ratio
 *    of arithmetic instructions due to reduced global memory traffic.
 *
 * No attempt is made to compute the instruction mix programmatically,
 * because that information is only available via the profiler after
 * the kernel execution.  The program's purpose is to provide a
 * suitable workload for such profiling.
 ********************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 1024          // Matrix dimension (N x N)
#define TILE_SIZE 16    // Tile width for tiled kernel

/* ------------------------------------------------------------------
 * Kernel: Naive matrix multiplication
 * Each thread computes one element of the output matrix C
 * ------------------------------------------------------------------ */
__global__ void matMulNaive(const float *A, const float *B, float *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index of C
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index of C

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

/* ------------------------------------------------------------------
 * Kernel: Tiled matrix multiplication using shared memory
 * ------------------------------------------------------------------ */
__global__ void matMulTiled(const float *A, const float *B, float *C, int n)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile of A into shared memory
        int tiledRow = row;
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        if (tiledRow < n && tiledCol < n)
            As[threadIdx.y][threadIdx.x] = A[tiledRow * n + tiledCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B into shared memory
        tiledRow = t * TILE_SIZE + threadIdx.y;
        tiledCol = col;
        if (tiledRow < n && tiledCol < n)
            Bs[threadIdx.y][threadIdx.x] = B[tiledRow * n + tiledCol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the two tiles together
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

/* ------------------------------------------------------------------
 * Helper: Check for CUDA errors
 * ------------------------------------------------------------------ */
void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%s)\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* ------------------------------------------------------------------
 * Helper: Simple matrix multiplication on CPU for verification
 * ------------------------------------------------------------------ */
void cpuMatMul(const float *A, const float *B, float *C, int n)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
}

/* ------------------------------------------------------------------
 * Main function
 * ------------------------------------------------------------------ */
int main()
{
    int n = N;
    size_t bytes = n * n * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_naive = (float*)malloc(bytes);
    float *h_C_tiled = (float*)malloc(bytes);
    float *h_C_cpu  = (float*)malloc(bytes);

    // Initialize input matrices with random data
    for (int i = 0; i < n * n; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");
    checkCuda(cudaMalloc(&d_B, bytes), "cudaMalloc d_B");
    checkCuda(cudaMalloc(&d_C, bytes), "cudaMalloc d_C");

    // Copy data to device
    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "Memcpy A");
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "Memcpy B");

    // Configure execution parameters
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                 (n + blockDim.y - 1) / blockDim.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop),  "cudaEventCreate stop");

    /* ------------------ Naive kernel ------------------ */
    checkCuda(cudaEventRecord(start), "Event record start naive");
    matMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    checkCuda(cudaEventRecord(stop), "Event record stop naive");
    checkCuda(cudaEventSynchronize(stop), "Event synchronize naive");

    // Copy result back
    checkCuda(cudaMemcpy(h_C_naive, d_C, bytes, cudaMemcpyDeviceToHost),
              "Memcpy result naive");

    float ms_naive = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms_naive, start, stop), "Elapsed time naive");
    printf("Naive kernel time: %f ms\n", ms_naive);

    /* ------------------ Tiled kernel ------------------ */
    checkCuda(cudaEventRecord(start), "Event record start tiled");
    matMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    checkCuda(cudaEventRecord(stop), "Event record stop tiled");
    checkCuda(cudaEventSynchronize(stop), "Event synchronize tiled");

    // Copy result back
    checkCuda(cudaMemcpy(h_C_tiled, d_C, bytes, cudaMemcpyDeviceToHost),
              "Memcpy result tiled");

    float ms_tiled = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms_tiled, start, stop), "Elapsed time tiled");
    printf("Tiled kernel time: %f ms\n", ms_tiled);

    /* ------------------ CPU reference ------------------ */
    cpuMatMul(h_A, h_B, h_C_cpu, n);

    // Verify correctness (optional, can be skipped for large n)
    bool ok = true;
    for (int i = 0; i < n * n && ok; ++i) {
        if (fabs(h_C_naive[i] - h_C_cpu[i]) > 1e-3f ||
            fabs(h_C_tiled[i] - h_C_cpu[i]) > 1e-3f) {
            ok = false;
        }
    }
    printf("Verification: %s\n", ok ? "PASSED" : "FAILED");

    // Clean up
    checkCuda(cudaEventDestroy(start), "Event destroy start");
    checkCuda(cudaEventDestroy(stop),  "Event destroy stop");
    checkCuda(cudaFree(d_A), "cudaFree d_A");
    checkCuda(cudaFree(d_B), "cudaFree d_B");
    checkCuda(cudaFree(d_C), "cudaFree d_C");
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tiled); free(h_C_cpu);

    return 0;
}
```