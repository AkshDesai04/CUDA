```cuda
/*
Aim of the program:
Use `nsight compute` or `nvprof` to compare the global memory traffic between the two versions. The tiled version should have significantly less traffic.

Thinking:
1. I want to demonstrate how global memory traffic can differ between a naive implementation of matrix multiplication and a tiled (shared memory) implementation.  
2. The program will contain two kernels:
   - `naiveMatMul`: each thread reads the entire i-th row and j-th column from global memory to compute one element of C. This leads to many redundant global memory loads.
   - `tiledMatMul`: uses shared memory to load tiles of A and B once and reuse them for all threads in a block, thereby reducing global memory traffic.
3. The host code will:
   - Allocate and initialize two input matrices A and B with deterministic values (e.g., A[i][j] = i + j, B[i][j] = i - j) so we can verify correctness.
   - Allocate device memory, copy inputs to device, launch each kernel, copy back the results.
   - Measure elapsed execution time for each kernel using CUDA events (this is not traffic measurement, but useful for reference).
   - Verify that the outputs from both kernels are equal (within a tolerance).
4. The user will run the binary under Nsight Compute or nvprof and look at the reported global memory traffic for each kernel. They should observe that the tiled kernel reports significantly fewer bytes transferred compared to the naive one.
5. The code uses simple error checking macros, and is self‑contained in a single `.cu` file, ready to compile with `nvcc`.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cassert>

#define CHECK_CUDA(call)                                                     \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__     \
                      << " (" << cudaGetErrorString(err) << ")\n";          \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    }

const int N = 1024;            // Size of matrices (NxN)
const int TILE_DIM = 32;       // Tile width for tiled kernel

// Naive matrix multiplication kernel
__global__ void naiveMatMul(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // column index

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            // Each thread reads A[row*n + k] and B[k*n + col] from global memory
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Tiled matrix multiplication kernel using shared memory
__global__ void tiledMatMul(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* C,
                            int n) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0f;

    // Loop over tiles of A and B
    for (int t = 0; t < (n + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Load tile of A into shared memory
        if (row < n && t * TILE_DIM + threadIdx.x < n)
            As[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_DIM + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B into shared memory
        if (t * TILE_DIM + threadIdx.y < n && col < n)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial sum for this tile
        for (int k = 0; k < TILE_DIM; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    // Write result to global memory
    if (row < n && col < n)
        C[row * n + col] = sum;
}

// Helper to initialize matrix with simple pattern
void initMatrix(float* mat, int n, bool isA) {
    for (int i = 0; i < n * n; ++i) {
        int row = i / n;
        int col = i % n;
        if (isA)
            mat[i] = static_cast<float>(row + col);
        else
            mat[i] = static_cast<float>(row - col);
    }
}

// Verify that two matrices are approximately equal
bool verify(const float* A, const float* B, int n) {
    const float eps = 1e-3f;
    for (int i = 0; i < n * n; ++i) {
        if (fabs(A[i] - B[i]) > eps) {
            std::cerr << "Mismatch at index " << i << ": " << A[i] << " vs " << B[i] << "\n";
            return false;
        }
    }
    return true;
}

int main() {
    const size_t size = N * N * sizeof(float);

    // Host matrices
    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C_naive = new float[N * N];
    float* h_C_tiled = new float[N * N];

    // Initialize inputs
    initMatrix(h_A, N, true);
    initMatrix(h_B, N, false);

    // Device matrices
    float *d_A, *d_B, *d_C_naive, *d_C_tiled;
    CHECK_CUDA(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA(cudaMalloc((void**)&d_C_naive, size));
    CHECK_CUDA(cudaMalloc((void**)&d_C_tiled, size));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    // Events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Naive kernel
    CHECK_CUDA(cudaEventRecord(start));
    naiveMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C_naive, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_naive;
    CHECK_CUDA(cudaEventElapsedTime(&ms_naive, start, stop));

    // Tiled kernel
    CHECK_CUDA(cudaEventRecord(start));
    tiledMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C_tiled, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_tiled;
    CHECK_CUDA(cudaEventElapsedTime(&ms_tiled, start, stop));

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_C_naive, d_C_naive, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_tiled, d_C_tiled, size, cudaMemcpyDeviceToHost));

    // Verify correctness
    bool ok_naive = verify(h_C_naive, h_C_tiled, N);
    std::cout << "Verification: " << (ok_naive ? "PASS" : "FAIL") << "\n";

    std::cout << "Naive kernel time: " << ms_naive << " ms\n";
    std::cout << "Tiled  kernel time: " << ms_tiled << " ms\n";

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_naive;
    delete[] h_C_tiled;
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C_naive));
    CHECK_CUDA(cudaFree(d_C_tiled));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
```