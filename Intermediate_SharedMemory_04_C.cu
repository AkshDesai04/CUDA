```cuda
/*
Aim of the program:
Plot the GFLOPS of both kernels for a range of matrix sizes (e.g., 256, 512, 1024, 2048).

Thought Process:
1. The goal is to benchmark two CUDA kernels – a naive matrix multiplication kernel
   (each thread computes one output element using global memory only) and a tiled
   matrix multiplication kernel that uses shared memory to reduce global memory
   traffic. The program will measure the execution time of each kernel for a set
   of square matrix sizes (256, 512, 1024, 2048) and compute the achieved
   performance in GFLOPS.
2. GFLOPS for matrix multiplication is calculated as:
      FLOPS = 2 * N^3
   where N is the dimension of the square matrices. The performance in
   gigaflops (GFLOPS) is then:
      GFLOPS = (2 * N^3) / (time_in_seconds * 1e9)
   or using milliseconds:
      GFLOPS = (2 * N^3) / (time_in_ms / 1000) / 1e9
3. The program will:
   - Allocate host memory for matrices A, B, and C.
   - Allocate device memory for the same.
   - Initialize A and B with random float values.
   - For each matrix size:
     * Copy data to device.
     * Launch the naive kernel and time it.
     * Launch the tiled kernel and time it.
     * Compute GFLOPS for both kernels.
     * Print results in a table format that can be easily redirected to a CSV
       or plotted using external tools like Gnuplot.
4. Kernels:
   - `matMulNaive`: Each thread computes one element C[i][j] using
     a simple loop over k, accessing global memory for A and B.
   - `matMulTiled`: Each thread block loads a tile of A and B into shared
     memory, performs partial multiplication, and accumulates the result.
     The block size is set to 16x16 for demonstration; it can be tuned.
5. Timing uses CUDA events (`cudaEventRecord` / `cudaEventElapsedTime`) which
   provide sub-millisecond resolution on most GPUs.
6. Error checking is added via a macro `CUDA_CHECK` to make debugging easier.
7. After the benchmark, all allocated memory is freed, and the program exits.

The output will look like:
```
Size,Naive_GFLOPS,Tiled_GFLOPS
256,123.45,234.56
512,...
```
which can be plotted externally. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 16  // For tiled kernel

// Error checking macro
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Naive matrix multiplication kernel
__global__ void matMulNaive(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // i
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // j
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled matrix multiplication kernel using shared memory
__global__ void matMulTiled(const float *A, const float *B, float *C, int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Load tile into shared memory
        int tiledRow = row;
        int tiledCol = t * BLOCK_SIZE + threadIdx.x;
        if (tiledRow < N && tiledCol < N)
            As[threadIdx.y][threadIdx.x] = A[tiledRow * N + tiledCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        tiledRow = t * BLOCK_SIZE + threadIdx.y;
        tiledCol = col;
        if (tiledRow < N && tiledCol < N)
            Bs[threadIdx.y][threadIdx.x] = B[tiledRow * N + tiledCol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Function to fill matrix with random floats
void initMatrix(float *mat, int N) {
    for (int i = 0; i < N * N; ++i) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// Compute GFLOPS: 2*N^3 operations
double computeGFLOPS(int N, float milliseconds) {
    double ops = 2.0 * N * N * N;   // 2*N^3
    double seconds = milliseconds / 1000.0;
    return ops / seconds / 1e9;     // GFLOPS
}

int main() {
    // Seed RNG
    srand((unsigned)time(NULL));

    // Matrix sizes to benchmark
    const int sizes[] = {256, 512, 1024, 2048};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("Size,Naive_GFLOPS,Tiled_GFLOPS\n");

    for (int s = 0; s < num_sizes; ++s) {
        int N = sizes[s];
        size_t bytes = N * N * sizeof(float);

        // Allocate host memory
        float *h_A = (float*)malloc(bytes);
        float *h_B = (float*)malloc(bytes);
        float *h_C = (float*)malloc(bytes);  // Not used for verification

        initMatrix(h_A, N);
        initMatrix(h_B, N);

        // Allocate device memory
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

        // Copy inputs to device
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

        // Define block and grid sizes
        dim3 blockDim(16, 16);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                     (N + blockDim.y - 1) / blockDim.y);

        // Events for timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // ---------- Naive Kernel ----------
        CUDA_CHECK(cudaEventRecord(start, 0));
        matMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms_naive = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));

        // ---------- Tiled Kernel ----------
        CUDA_CHECK(cudaEventRecord(start, 0));
        matMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms_tiled = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_tiled, start, stop));

        // Compute GFLOPS
        double gflops_naive = computeGFLOPS(N, ms_naive);
        double gflops_tiled = computeGFLOPS(N, ms_tiled);

        // Output results
        printf("%d,%.3f,%.3f\n", N, gflops_naive, gflops_tiled);

        // Cleanup
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        free(h_A);
        free(h_B);
        free(h_C);
    }

    return 0;
}
```