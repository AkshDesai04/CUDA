/*
Aim of the program: Report the execution time speedup of the optimized version over the naive one.

Thinking:
- To measure execution time speedup, we need two implementations of the same computation: a naive version and an optimized version.
- Matrix multiplication is a common GPU benchmark where a naive kernel (one thread per output element with global memory accesses) can be compared against a tiled kernel that uses shared memory.
- We will implement both kernels for square matrices of size N x N (using N=512 for a reasonable workload).
- We'll allocate matrices A, B, and C on the device, initialize A and B with random data on the host, copy them to the device.
- We'll time each kernel execution using cudaEventRecord and cudaEventElapsedTime, which gives elapsed time in milliseconds.
- For stability, each kernel will be launched only once, but you could loop several times if desired.
- After timing, we compute speedup = naive_time / optimized_time and print the results.
- We include simple error checking after CUDA API calls.
- The program uses double precision for demonstration; you could switch to float if preferred.
- The code is self-contained and compiles with nvcc. It prints naive time, optimized time, and speedup.
*/

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

// Error checking macro
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__     \
                      << " : " << cudaGetErrorString(err) << std::endl;      \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

const int N = 512;               // Matrix dimension
const int TILE_WIDTH = 16;       // Tile size for shared memory kernel

// Naive matrix multiplication kernel
__global__ void matMulNaive(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // column index

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Optimized matrix multiplication using shared memory tiling
__global__ void matMulShared(const double* A, const double* B, double* C, int n) {
    __shared__ double As[TILE_WIDTH][TILE_WIDTH];
    __shared__ double Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    double sum = 0.0;
    for (int m = 0; m < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        // Load tiles into shared memory
        if (row < n && (m * TILE_WIDTH + threadIdx.x) < n)
            As[threadIdx.y][threadIdx.x] = A[row * n + m * TILE_WIDTH + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;

        if (col < n && (m * TILE_WIDTH + threadIdx.y) < n)
            Bs[threadIdx.y][threadIdx.x] = B[(m * TILE_WIDTH + threadIdx.y) * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

int main() {
    size_t bytes = N * N * sizeof(double);

    // Host allocations
    double *h_A = (double*)malloc(bytes);
    double *h_B = (double*)malloc(bytes);
    double *h_C = (double*)malloc(bytes); // Not used, but allocated for completeness

    // Initialize matrices with random data
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<double>(std::rand()) / RAND_MAX;
        h_B[i] = static_cast<double>(std::rand()) / RAND_MAX;
    }

    // Device allocations
    double *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Create events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ---------- Naive kernel ----------
    CHECK_CUDA(cudaEventRecord(start));
    matMulNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float timeNaive;
    CHECK_CUDA(cudaEventElapsedTime(&timeNaive, start, stop));

    // ---------- Optimized kernel ----------
    CHECK_CUDA(cudaEventRecord(start));
    matMulShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float timeOptimized;
    CHECK_CUDA(cudaEventElapsedTime(&timeOptimized, start, stop));

    // Compute speedup
    double speedup = timeNaive / timeOptimized;

    // Output results
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Naive kernel time: " << timeNaive << " ms" << std::endl;
    std::cout << "Optimized kernel time: " << timeOptimized << " ms" << std::endl;
    std::cout << "Speedup (naive / optimized): " << speedup << "x" << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
