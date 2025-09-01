```c
/*
 * Aim of the program:
 * Profile for different tile sizes (e.g., 8x8, 16x16, 32x32) and find the optimal one for your GPU.
 *
 * Thinking:
 * The goal is to benchmark a tiled matrix multiplication kernel on a CUDA device
 * using different block (tile) sizes.  We choose square tiles of sizes 8, 16, and 32,
 * which are common tile sizes that map to different warp configurations
 * (8x8 = 64 threads, 16x16 = 256 threads, 32x32 = 1024 threads).
 *
 * The kernel uses shared memory to load tiles of A and B matrices.
 * To keep the kernel flexible, we use dynamic shared memory and rely on
 * blockDim.x (assumed equal to blockDim.y) to specify the tile size at launch time.
 *
 * The host code:
 * 1. Generates two random N×N matrices (float).
 * 2. Copies them to device memory.
 * 3. For each tile size:
 *    - Sets up grid and block dimensions.
 *    - Computes the shared memory size needed.
 *    - Launches the kernel and times the execution with cudaEvent.
 * 4. Reports the elapsed time for each tile size and identifies the fastest one.
 *
 * We include a simple CUDA error checking macro for clarity.
 * The program is self‑contained and can be compiled with:
 *     nvcc -O2 -arch=sm_70 -o prof_matrix_mul prof_matrix_mul.cu
 * and run with: ./prof_matrix_mul
 *
 * Note:
 * - The matrix size N is set to 1024 for a reasonable benchmark.
 * - For simplicity, we skip the optional correctness check; however
 *   the kernel produces the same result for all tile sizes, so the
 *   timing comparison is meaningful.
 */

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

/* CUDA error checking macro */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

/* Tiled matrix multiplication kernel using dynamic shared memory */
__global__ void matMulKernel(const float *A, const float *B, float *C, int N)
{
    /* Determine tile size from block dimensions */
    int tileSize = blockDim.x;   // Assuming square blocks (blockDim.x == blockDim.y)

    /* Shared memory allocation: two tiles A and B */
    extern __shared__ float sharedMem[];
    float *tileA = sharedMem;
    float *tileB = sharedMem + tileSize * tileSize;

    /* Global row and column this thread is responsible for */
    int row = blockIdx.y * tileSize + threadIdx.y;
    int col = blockIdx.x * tileSize + threadIdx.x;

    float acc = 0.0f;   // Accumulator for the output element

    /* Loop over tiles */
    for (int t = 0; t < (N + tileSize - 1) / tileSize; ++t)
    {
        /* Load tile of A into shared memory */
        int Arow = row;
        int Acol = t * tileSize + threadIdx.x;
        if (Arow < N && Acol < N)
            tileA[threadIdx.y * tileSize + threadIdx.x] = A[Arow * N + Acol];
        else
            tileA[threadIdx.y * tileSize + threadIdx.x] = 0.0f;

        /* Load tile of B into shared memory */
        int Brow = t * tileSize + threadIdx.y;
        int Bcol = col;
        if (Brow < N && Bcol < N)
            tileB[threadIdx.y * tileSize + threadIdx.x] = B[Brow * N + Bcol];
        else
            tileB[threadIdx.y * tileSize + threadIdx.x] = 0.0f;

        __syncthreads();

        /* Compute partial sum for this tile */
        for (int k = 0; k < tileSize; ++k)
            acc += tileA[threadIdx.y * tileSize + k] * tileB[k * tileSize + threadIdx.x];

        __syncthreads();
    }

    /* Write result to global memory */
    if (row < N && col < N)
        C[row * N + col] = acc;
}

/* Helper to generate random float matrices */
void initMatrix(float *mat, int N)
{
    for (int i = 0; i < N * N; ++i)
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
}

int main()
{
    srand(static_cast<unsigned>(time(nullptr)));

    const int N = 1024;  // Matrix dimension
    const int TILE_SIZES[] = {8, 16, 32};
    const int NUM_TILE_SIZES = sizeof(TILE_SIZES) / sizeof(TILE_SIZES[0]);

    /* Host memory allocation */
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);  // Not used for correctness, but to receive output

    /* Initialize host matrices */
    initMatrix(h_A, N);
    initMatrix(h_B, N);

    /* Device memory allocation */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    /* Copy input matrices to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    float bestTime = 1e30f;
    int bestTileSize = TILE_SIZES[0];

    printf("Matrix size: %d x %d\n", N, N);
    printf("Profiling tile sizes...\n\n");

    /* Iterate over tile sizes */
    for (int idx = 0; idx < NUM_TILE_SIZES; ++idx)
    {
        int tileSize = TILE_SIZES[idx];

        dim3 blockDim(tileSize, tileSize);
        dim3 gridDim((N + tileSize - 1) / tileSize,
                     (N + tileSize - 1) / tileSize);

        size_t sharedMemBytes = 2 * tileSize * tileSize * sizeof(float);

        /* Timing events */
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        /* Warm-up launch */
        matMulKernel<<<gridDim, blockDim, sharedMemBytes>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaGetLastError());

        /* Record start */
        CUDA_CHECK(cudaEventRecord(start, 0));

        /* Kernel launch */
        matMulKernel<<<gridDim, blockDim, sharedMemBytes>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaGetLastError());

        /* Record stop */
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        /* Compute elapsed time */
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        printf("Tile size %dx%d: %.3f ms\n", tileSize, tileSize, milliseconds);

        if (milliseconds < bestTime) {
            bestTime = milliseconds;
            bestTileSize = tileSize;
        }

        /* Clean up events */
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    printf("\nOptimal tile size: %dx%d with %.3f ms\n", bestTileSize, bestTileSize, bestTime);

    /* Free device memory */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    /* Free host memory */
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```