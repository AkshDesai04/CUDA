```cpp
/* 
Aim of the program:
Modify the tiled matrix multiplication to handle matrix dimensions that are not perfect multiples of the tile size. Threads that would read past the matrix boundary should instead load a 0 into the shared memory tile.

Thinking:
- Implement a standard CUDA tiled matrix multiplication kernel where each thread block computes a TILE_SIZE x TILE_SIZE submatrix of the result.
- Because matrix dimensions may not be divisible by TILE_SIZE, we must guard against out‑of‑bounds accesses when loading tiles from A and B into shared memory. 
- Each thread loads its element into shared memory if the global index is within bounds; otherwise it writes 0.
- After loading, __syncthreads() ensures all loads are finished before the multiplication loop.
- We loop over the K dimension in TILE_SIZE chunks. For each chunk we load a tile of A (rows fixed by blockIdx.y*tile, columns advanced by the tile index) and a tile of B (rows advanced, columns fixed by blockIdx.x*tile). The same bounds checks apply.
- Accumulate the product of the two tiles into a local variable `sum`. Once all tiles are processed, write the result to C if the global position is valid.
- Host code allocates matrices of arbitrary size, initializes them with random floats, copies them to the GPU, launches the kernel, copies the result back, and verifies against a CPU implementation.
- Error checking is added via a macro to keep the code clean.
*/

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// Macro for CUDA error checking
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
                    cudaGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// CUDA kernel: tiled matrix multiplication with bounds checking
__global__ void matMulKernel(const float *A, const float *B, float *C,
                             int M, int N, int K)
{
    // Shared memory tiles for A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Row and column index of the element to compute
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Accumulator
    float sum = 0.0f;

    // Number of tiles along the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t)
    {
        // Global indices for elements to load into shared memory
        int aRow = row;
        int aCol = t * TILE_SIZE + threadIdx.x; // column in A

        int bRow = t * TILE_SIZE + threadIdx.y; // row in B
        int bCol = col;

        // Load element of A into shared memory (with bounds check)
        if (aRow < M && aCol < K)
            As[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load element of B into shared memory (with bounds check)
        if (bRow < K && bCol < N)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Multiply the two tiles together
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the result (with bounds check)
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// CPU reference implementation for verification
void matMulCPU(const float *A, const float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

int main()
{
    // Example matrix dimensions (not multiples of TILE_SIZE)
    int M = 50; // rows of A and C
    int K = 64; // columns of A and rows of B
    int N = 70; // columns of B and C

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);
    float *h_C_ref = (float*)malloc(sizeC);

    // Initialize matrices with random values
    srand((unsigned)time(NULL));
    for (int i = 0; i < M * K; ++i)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeA));
    CHECK_CUDA(cudaMalloc((void**)&d_B, sizeB));
    CHECK_CUDA(cudaMalloc((void**)&d_C, sizeC));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Verify result
    matMulCPU(h_A, h_B, h_C_ref, M, N, K);

    // Compute error
    double maxError = 0.0;
    for (int i = 0; i < M * N; ++i)
    {
        double err = fabs(h_C[i] - h_C_ref[i]);
        if (err > maxError) maxError = err;
    }

    printf("Max error: %e\n", maxError);

    // Free memory
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
```