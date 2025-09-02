```cpp
/*
Aim: The kernel will have an outer loop that iterates through the tiles of A and B needed to compute one tile of C.

Thinking:
- We want to implement tiled matrix multiplication using CUDA. The kernel will load sub-matrices (tiles) of A and B into shared memory.
- Each block is responsible for computing a TILE_SIZE x TILE_SIZE sub-matrix of the result C. The block dimensions match the tile size.
- An outer loop iterates over the number of tiles along the K dimension (the shared dimension of A and B). In each iteration, a tile of A (from the current column of tiles) and a tile of B (from the current row of tiles) are loaded into shared memory.
- After loading, threads synchronize and compute partial products for the current tile. This is repeated until all tiles have been processed. The sum accumulates the contributions from all tiles.
- For simplicity we assume that the matrix dimension N is a multiple of TILE_SIZE. The code can be extended to handle padding if needed.
- The host code allocates memory, initializes matrices, copies data to the device, launches the kernel, and copies the result back.
- Basic error checking is performed. For large matrices, we also time the kernel execution and optionally compare against a CPU implementation for correctness.
- The program accepts the matrix dimension as a command line argument. If omitted, a default size (e.g., 1024) is used.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>
#include <iostream>

// Tile size (adjust as needed)
#define TILE_SIZE 16

// CUDA error checking macro
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",                 \
                    __FILE__, __LINE__, err, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

// Kernel: tiled matrix multiplication
__global__ void matMulTiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int N)
{
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Identify row and column of the element to compute
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Number of tiles to iterate over along K dimension
    int numTiles = N / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t)
    {
        // Load tile of A into shared memory
        int Arow = row;
        int Acol = t * TILE_SIZE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = A[Arow * N + Acol];

        // Load tile of B into shared memory
        int Brow = t * TILE_SIZE + threadIdx.y;
        int Bcol = col;
        Bs[threadIdx.y][threadIdx.x] = B[Brow * N + Bcol];

        // Wait until all threads have loaded their tiles
        __syncthreads();

        // Multiply the two tiles together
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Wait until all threads have finished computing with the tile
        __syncthreads();
    }

    // Write the computed value to the output matrix
    C[row * N + col] = sum;
}

// Utility function to initialize matrix with random values
void initMatrix(float* mat, int N)
{
    for (int i = 0; i < N * N; ++i)
    {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// CPU implementation for verification (optional)
void matMulCPU(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char* argv[])
{
    // Parse matrix size from command line (default to 1024)
    int N = 1024;
    if (argc > 1)
    {
        N = atoi(argv[1]);
        if (N <= 0)
        {
            std::cerr << "Matrix size must be positive." << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Ensure N is a multiple of TILE_SIZE for simplicity
    if (N % TILE_SIZE != 0)
    {
        std::cerr << "Matrix size must be a multiple of TILE_SIZE (" << TILE_SIZE << ")." << std::endl;
        return EXIT_FAILURE;
    }

    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    float* h_C_ref = (float*)malloc(size); // For CPU reference

    if (!h_A || !h_B || !h_C || !h_C_ref)
    {
        std::cerr << "Failed to allocate host memory." << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize input matrices
    srand(static_cast<unsigned>(time(nullptr)));
    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid(N / TILE_SIZE, N / TILE_SIZE);

    // Launch kernel and time it
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    matMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "GPU matrix multiplication took " << milliseconds << " ms." << std::endl;

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Optional CPU verification for small matrices
    if (N <= 512)
    {
        std::cout << "Computing reference result on CPU..." << std::endl;
        matMulCPU(h_A, h_B, h_C_ref, N);

        // Verify results
        bool match = true;
        const float eps = 1e-4f;
        for (int i = 0; i < N * N; ++i)
        {
            if (fabs(h_C[i] - h_C_ref[i]) > eps)
            {
                std::cerr << "Mismatch at index " << i
                          << ": GPU=" << h_C[i]
                          << ", CPU=" << h_C_ref[i] << std::endl;
                match = false;
                break;
            }
        }
        std::cout << (match ? "Results match!" : "Results differ.") << std::endl;
    }

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