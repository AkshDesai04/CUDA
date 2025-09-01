/*
Aim: The kernel will have an outer loop that iterates through the tiles of A and B needed to compute one tile of C.

Thinking:
The goal is to implement a classic tiled matrix multiplication on the GPU, where each CUDA block computes a sub‑matrix (tile) of the result matrix C.  
- Each thread within a block is responsible for computing a single element of the output tile.  
- To reduce global memory traffic, we load sub‑matrices of A and B into fast shared memory.  
- The outer loop iterates over the number of tiles that cover the shared dimension (K). In each iteration, the block loads a new tile of A and a new tile of B, multiplies the two tiles in shared memory, and accumulates the partial result.  
- After all tiles have been processed, each thread writes its final value to global memory.  
Key design points:  
1. **Block and grid dimensions**: We choose a fixed TILE_WIDTH (e.g., 16 or 32) that matches the GPU’s warp size. The grid size is computed so that all rows and columns of C are covered.  
2. **Shared memory layout**: Two square tiles of size TILE_WIDTH x TILE_WIDTH are stored contiguously in shared memory, one for A and one for B.  
3. **Indexing and bounds checking**: For non‑square matrices or sizes not divisible by TILE_WIDTH, threads that map outside the matrix bounds simply contribute zero.  
4. **Synchronization**: `__syncthreads()` is used after loading into shared memory and after the inner multiplication loop to ensure all data is ready before it is read.  
5. **Memory footprint**: The shared memory requirement per block is `2 * TILE_WIDTH * TILE_WIDTH * sizeof(float)`.  
6. **Host code**: The host allocates matrices A, B, and C, copies them to device memory, launches the kernel, and copies the result back. Error checking is performed via a macro.  
This structure yields efficient memory access patterns and high throughput on modern GPUs. The code below is a complete, self‑contained .cu file that demonstrates this approach with a small test case.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

#define TILE_WIDTH 16

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call)                                                     \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in file '" << __FILE__                  \
                      << "' in line " << __LINE__ << ": "                   \
                      << cudaGetErrorString(err) << std::endl;               \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    }

// Kernel: tiled matrix multiplication
__global__ void MatMulTiled(const float *A, const float *B, float *C,
                            int M, int N, int K)
{
    // Shared memory for tiles of A and B
    extern __shared__ float sharedMem[];
    float *As = sharedMem;                         // TILE_WIDTH * TILE_WIDTH
    float *Bs = sharedMem + TILE_WIDTH * TILE_WIDTH; // TILE_WIDTH * TILE_WIDTH

    // Identify row and column of the element in C to compute
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float Cvalue = 0.0f;

    // Loop over all tiles needed for this element
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t)
    {
        // Load tile of A into shared memory
        int aRow = row;
        int aCol = t * TILE_WIDTH + threadIdx.x;
        if (aRow < M && aCol < K)
            As[threadIdx.y * TILE_WIDTH + threadIdx.x] = A[aRow * K + aCol];
        else
            As[threadIdx.y * TILE_WIDTH + threadIdx.x] = 0.0f;

        // Load tile of B into shared memory
        int bRow = t * TILE_WIDTH + threadIdx.y;
        int bCol = col;
        if (bRow < K && bCol < N)
            Bs[threadIdx.y * TILE_WIDTH + threadIdx.x] = B[bRow * N + bCol];
        else
            Bs[threadIdx.y * TILE_WIDTH + threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the two tiles together
        for (int k = 0; k < TILE_WIDTH; ++k)
            Cvalue += As[threadIdx.y * TILE_WIDTH + k] *
                      Bs[k * TILE_WIDTH + threadIdx.x];

        __syncthreads();
    }

    // Write the result to global memory
    if (row < M && col < N)
        C[row * N + col] = Cvalue;
}

// Helper function to initialize matrix with random floats
void initMatrix(float *mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; ++i)
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
}

// Helper function to print matrix
void printMatrix(const float *mat, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
            std::cout << mat[i * cols + j] << " ";
        std::cout << std::endl;
    }
}

int main()
{
    srand(static_cast<unsigned>(time(nullptr)));

    // Matrix dimensions
    int M = 64; // rows of A and C
    int K = 128; // columns of A and rows of B
    int N = 96; // columns of B and C

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);
    float *h_C_ref = (float *)malloc(sizeC);

    // Initialize host matrices
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    // Compute reference result on CPU
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += h_A[i * K + k] * h_B[k * N + j];
            h_C_ref[i * N + j] = sum;
        }
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void **)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void **)&d_C, sizeC));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH,
                 (M + TILE_WIDTH - 1) / TILE_WIDTH);
    size_t sharedMemSize = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);

    MatMulTiled<<<dimGrid, dimBlock, sharedMemSize>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Verify correctness
    bool match = true;
    for (int i = 0; i < M * N; ++i)
    {
        float diff = fabs(h_C[i] - h_C_ref[i]);
        if (diff > 1e-3f)
        {
            match = false;
            std::cerr << "Mismatch at index " << i
                      << ": GPU " << h_C[i]
                      << " CPU " << h_C_ref[i]
                      << " diff " << diff << std::endl;
            break;
        }
    }
    if (match)
        std::cout << "Result matches reference." << std::endl;
    else
        std::cout << "Result does NOT match reference." << std::endl;

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
