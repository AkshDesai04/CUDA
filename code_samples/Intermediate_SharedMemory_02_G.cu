```cpp
/*
 * Aim of the program:
 * Try using a different tile size, like 32x32, and see how it affects performance. 
 * Note that this requires more shared memory and registers.
 *
 * Thinking process:
 * 1. The user requests a CUDA program that explores the impact of a 32x32 tile size on performance.
 * 2. A tiled matrix multiplication is a canonical example where tile size directly influences
 *    shared memory usage, register pressure, and occupancy.
 * 3. I will write a self‑contained .cu file that:
 *    - Allocates two input matrices A (MxK) and B (KxN) and an output matrix C (MxN) in host memory.
 *    - Copies them to device memory.
 *    - Launches a kernel that performs matrix multiplication using a 32x32 tile,
 *      loading each tile into shared memory.
 *    - Measures kernel execution time with cudaEvent.
 *    - Computes throughput (GFlops) and prints a few output elements for verification.
 * 4. The kernel uses 32x32 thread blocks, each thread computing one element of the
 *    output block. Shared memory arrays are statically sized [32][32] for A and B.
 * 5. The code handles arbitrary matrix dimensions, using bounds checks for incomplete
 *    tiles on the edges.
 * 6. Host code parses optional command‑line arguments for matrix dimensions, defaults
 *    to 1024x1024x1024 if not provided.
 * 7. The program uses standard CUDA C API calls and can be compiled with nvcc.
 * 8. Finally, the output is only the .cu file content as requested.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define TILE_SIZE 32

// CUDA error checking macro
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

// Tiled matrix multiplication kernel
__global__ void matMulKernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* C,
                             int M, int N, int K)
{
    // Compute row and column index of the element to compute
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Shared memory for tiles of A and B
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    // Number of tiles along the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        // Load tile of A into shared memory
        int Arow = row;
        int Acol = t * TILE_SIZE + threadIdx.x;
        if (Arow < M && Acol < K)
            Asub[threadIdx.y][threadIdx.x] = A[Arow * K + Acol];
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B into shared memory
        int Brow = t * TILE_SIZE + threadIdx.y;
        int Bcol = col;
        if (Brow < K && Bcol < N)
            Bsub[threadIdx.y][threadIdx.x] = B[Brow * N + Bcol];
        else
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the two tiles together
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// CPU implementation for verification
void matMulCPU(const float* A, const float* B, float* C,
               int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char* argv[])
{
    // Matrix dimensions (default 1024x1024x1024)
    int M = 1024; // rows of A and C
    int K = 1024; // columns of A, rows of B
    int N = 1024; // columns of B and C

    if (argc == 4) {
        M = atoi(argv[1]);
        K = atoi(argv[2]);
        N = atoi(argv[3]);
    } else {
        printf("Using default dimensions: M=%d, K=%d, N=%d\n", M, K, N);
        printf("To specify, run: %s M K N\n", argv[0]);
    }

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);
    float* h_C_ref = (float*)malloc(sizeC);

    // Initialize host matrices with random values
    srand(42);
    for (int i = 0; i < M * K; ++i) h_A[i] = (float)(rand() % 10);
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)(rand() % 10);
    for (int i = 0; i < M * N; ++i) h_C[i] = 0.0f;

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeC));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Launch kernel
    CUDA_CHECK(cudaEventRecord(start, 0));
    matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double seconds = ms / 1000.0;

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Compute reference result on CPU for verification
    printf("Computing reference result on CPU for verification (this may take a while)...\n");
    matMulCPU(h_A, h_B, h_C_ref, M, N, K);

    // Verify a few entries
    int errors = 0;
    for (int i = 0; i < 10 && i < M; ++i) {
        for (int j = 0; j < 10 && j < N; ++j) {
            float diff = fabs(h_C[i * N + j] - h_C_ref[i * N + j]);
            if (diff > 1e-3f) {
                printf("Mismatch at (%d,%d): GPU=%f, CPU=%f, diff=%f\n",
                       i, j, h_C[i * N + j], h_C_ref[i * N + j], diff);
                ++errors;
            }
        }
    }
    if (errors == 0) {
        printf("Verification PASSED for first 10x10 entries.\n");
    } else {
        printf("Verification FAILED: %d mismatches found.\n", errors);
    }

    // Compute GFLOPS
    double gflops = (2.0 * M * N * K) / (seconds * 1e9);
    printf("Kernel time: %f ms\n", ms);
    printf("Throughput: %.2f GFLOPS\n", gflops);

    // Clean up
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