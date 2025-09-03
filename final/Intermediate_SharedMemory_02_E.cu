/*
Aim: The kernel will have an outer loop that iterates through the tiles of A and B needed to compute one tile of C.

Thinking:
The requirement is to write a CUDA program that demonstrates matrix multiplication using tiling, where the kernel contains an outer loop that iterates over the tiles of the input matrices A and B to compute a single tile of the result matrix C.  
We choose a standard tiled matrix multiplication approach:  
- Each thread block computes a TILE_WIDTH × TILE_WIDTH tile of C.  
- Each thread in the block computes one element of that tile.  
- Shared memory is used to load tiles of A and B.  
- An outer loop iterates over the number of tiles along the shared dimension (k), loading a new tile of A and B each time, performing partial products, and accumulating into the partial sum for the element.  
- After the loop, the partial sum is written to global memory.  

The host code:
- Allocates and initializes two square matrices A and B (size N×N, float).  
- Copies them to device memory.  
- Launches the kernel with (N/TILE_WIDTH, N/TILE_WIDTH) blocks and TILE_WIDTH×TILE_WIDTH threads per block.  
- Copies the result back and optionally verifies correctness for small N.  
- Frees all allocated memory.  

This implementation keeps the code simple, focuses on the tiled kernel with the outer loop, and includes basic error checking. It does not handle arbitrary N that is not divisible by TILE_WIDTH; for brevity we assume N is a multiple of TILE_WIDTH. */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_WIDTH 16

// CUDA error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Kernel: tiled matrix multiplication
__global__ void MatMulTiled(const float *A, const float *B, float *C, int N)
{
    __shared__ float Asub[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bsub[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float Cvalue = 0.0f;

    // Outer loop over tiles of A and B
    for (int t = 0; t < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++t)
    {
        // Load tile of A
        int Arow = row;
        int Acol = t * TILE_WIDTH + threadIdx.x;
        if (Arow < N && Acol < N)
            Asub[threadIdx.y][threadIdx.x] = A[Arow * N + Acol];
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B
        int Brow = t * TILE_WIDTH + threadIdx.y;
        int Bcol = col;
        if (Brow < N && Bcol < N)
            Bsub[threadIdx.y][threadIdx.x] = B[Brow * N + Bcol];
        else
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the two tiles together
        for (int k = 0; k < TILE_WIDTH; ++k)
            Cvalue += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];

        __syncthreads();
    }

    // Write result
    if (row < N && col < N)
        C[row * N + col] = Cvalue;
}

// CPU reference implementation for verification
void MatMulCPU(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

int main(int argc, char *argv[])
{
    int N = 1024; // Default matrix size
    if (argc > 1)
        N = atoi(argv[1]);

    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_C_ref = (float *)malloc(size);

    // Initialize matrices with random values
    srand(time(NULL));
    for (int i = 0; i < N * N; ++i)
    {
        h_A[i] = (float)(rand() % 100) / 10.0f;
        h_B[i] = (float)(rand() % 100) / 10.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaCheckError(cudaMalloc((void **)&d_A, size));
    cudaCheckError(cudaMalloc((void **)&d_B, size));
    cudaCheckError(cudaMalloc((void **)&d_C, size));

    // Copy data to device
    cudaCheckError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    MatMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaCheckError(cudaDeviceSynchronize());

    // Copy result back to host
    cudaCheckError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify result for small N
    if (N <= 64)
    {
        MatMulCPU(h_A, h_B, h_C_ref, N);
        bool correct = true;
        for (int i = 0; i < N * N; ++i)
        {
            if (fabs(h_C[i] - h_C_ref[i]) > 1e-3f)
            {
                correct = false;
                printf("Mismatch at index %d: GPU %f, CPU %f\n", i, h_C[i], h_C_ref[i]);
                break;
            }
        }
        if (correct)
            printf("Verification PASSED for N = %d\n", N);
        else
            printf("Verification FAILED for N = %d\n", N);
    }

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
