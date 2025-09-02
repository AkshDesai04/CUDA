```cpp
/* 
Aim of the program:
Profile and compare the naive vs. tiled matrix multiplication kernels using CUDA events. Measure the execution time for a large matrix (e.g., 1024x1024) and report the speedup achieved by using shared memory.

Thinking:
- To compare the performance, two separate CUDA kernels are written: a naive kernel where each thread calculates a single element of the result matrix using a simple for-loop over the inner dimension, and a tiled (shared memory) kernel that loads submatrices (tiles) of the input matrices into shared memory to reduce global memory traffic.
- The matrices are square of size N = 1024 for a substantial workload.
- Random data is generated on the host to fill matrices A and B.
- CUDA events are used to time the kernels precisely; the events are recorded immediately before and after each kernel launch, and the elapsed time is calculated with cudaEventElapsedTime (in milliseconds).
- After execution, the result from the tiled kernel is compared against the naive kernel for correctness (checking a few entries).
- The speedup is computed as the ratio of the naive kernel time to the tiled kernel time.
- All host and device memory allocations, copies, and kernel launches include error checking.
- The code is fully self-contained and ready to compile with nvcc.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define N 1024           // Matrix size N x N
#define TILE_SIZE 16     // Tile width for shared memory kernel

// Check CUDA errors
inline void checkCuda(cudaError_t result, const char *msg)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

// Naive matrix multiplication kernel
__global__ void matMulNaive(const float *A, const float *B, float *C, int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index of C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // col index of C

    if (row < width && col < width)
    {
        float sum = 0.0f;
        for (int k = 0; k < width; ++k)
        {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Tiled matrix multiplication kernel using shared memory
__global__ void matMulTiled(const float *A, const float *B, float *C, int width)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (width + TILE_SIZE - 1) / TILE_SIZE; ++t)
    {
        // Load tiles into shared memory
        if (row < width && t * TILE_SIZE + threadIdx.x < width)
            As[threadIdx.y][threadIdx.x] = A[row * width + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < width && t * TILE_SIZE + threadIdx.y < width)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * width + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the two tiles
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < width && col < width)
        C[row * width + col] = sum;
}

int main()
{
    const int size = N * N;
    const size_t bytes = size * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_naive = (float *)malloc(bytes);
    float *h_C_tiled = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C_naive || !h_C_tiled)
    {
        fprintf(stderr, "Failed to allocate host matrices.\n");
        return EXIT_FAILURE;
    }

    // Initialize input matrices with random values
    for (int i = 0; i < size; ++i)
    {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C_naive, *d_C_tiled;
    checkCuda(cudaMalloc((void **)&d_A, bytes), "cudaMalloc d_A");
    checkCuda(cudaMalloc((void **)&d_B, bytes), "cudaMalloc d_B");
    checkCuda(cudaMalloc((void **)&d_C_naive, bytes), "cudaMalloc d_C_naive");
    checkCuda(cudaMalloc((void **)&d_C_tiled, bytes), "cudaMalloc d_C_tiled");

    // Copy input data to device
    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "Memcpy h_A -> d_A");
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "Memcpy h_B -> d_B");

    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);

    // CUDA events for timing
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "Event create start");
    checkCuda(cudaEventCreate(&stop), "Event create stop");

    // ----------------- Naive kernel timing -----------------
    checkCuda(cudaEventRecord(start), "Record start naive");
    matMulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C_naive, N);
    checkCuda(cudaEventRecord(stop), "Record stop naive");
    checkCuda(cudaEventSynchronize(stop), "Synchronize stop naive");

    float timeNaive = 0.0f;
    checkCuda(cudaEventElapsedTime(&timeNaive, start, stop), "Elapsed time naive");

    // ----------------- Tiled kernel timing -----------------
    checkCuda(cudaEventRecord(start), "Record start tiled");
    matMulTiled<<<gridSize, blockSize>>>(d_A, d_B, d_C_tiled, N);
    checkCuda(cudaEventRecord(stop), "Record stop tiled");
    checkCuda(cudaEventSynchronize(stop), "Synchronize stop tiled");

    float timeTiled = 0.0f;
    checkCuda(cudaEventElapsedTime(&timeTiled, start, stop), "Elapsed time tiled");

    // Copy results back to host
    checkCuda(cudaMemcpy(h_C_naive, d_C_naive, bytes, cudaMemcpyDeviceToHost), "Memcpy d_C_naive -> h_C_naive");
    checkCuda(cudaMemcpy(h_C_tiled, d_C_tiled, bytes, cudaMemcpyDeviceToHost), "Memcpy d_C_tiled -> h_C_tiled");

    // Verify a few elements
    bool correct = true;
    for (int i = 0; i < 5; ++i)
    {
        int idx = i * N + i;
        float diff = fabs(h_C_naive[idx] - h_C_tiled[idx]);
        if (diff > 1e-3f)
        {
            correct = false;
            break;
        }
    }
    if (correct)
        printf("Verification passed.\n");
    else
        printf("Verification failed.\n");

    // Compute speedup
    float speedup = timeNaive / timeTiled;

    printf("Naive kernel time: %f ms\n", timeNaive);
    printf("Tiled kernel time: %f ms\n", timeTiled);
    printf("Speedup (Naive / Tiled): %f\n", speedup);

    // Cleanup
    checkCuda(cudaEventDestroy(start), "Event destroy start");
    checkCuda(cudaEventDestroy(stop), "Event destroy stop");
    checkCuda(cudaFree(d_A), "cudaFree d_A");
    checkCuda(cudaFree(d_B), "cudaFree d_B");
    checkCuda(cudaFree(d_C_naive), "cudaFree d_C_naive");
    checkCuda(cudaFree(d_C_tiled), "cudaFree d_C_tiled");
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_tiled);

    return 0;
}
```