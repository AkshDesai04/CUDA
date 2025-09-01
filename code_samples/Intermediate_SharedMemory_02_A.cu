/*
Use a tile size of 16x16, meaning your thread block will also be 16x16.
This CUDA program implements a tiled matrix multiplication C = A × B using shared memory.
The tile size is set to 16×16, so each thread block is 16×16 threads. 
The kernel loads sub‑tiles of A and B into shared memory, performs partial products,
and accumulates the results. 
The host code allocates and initializes matrices on the CPU, copies them to the GPU,
launches the kernel, and copies the result back for verification. 
We use a square matrix of size N×N (default N=1024), but this can be changed.
We include basic error checking and simple timing for performance measurement.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define TILE_SIZE 16

/* CUDA error checking macro */
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/* Kernel for tiled matrix multiplication */
__global__ void matrixMulTiled(float *C, const float *A, const float *B, int width)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (width + TILE_SIZE - 1)/TILE_SIZE; ++t)
    {
        int Arow = row;
        int Acol = t * TILE_SIZE + threadIdx.x;
        int Brow = t * TILE_SIZE + threadIdx.y;
        int Bcol = col;

        // Load tiles into shared memory with bounds checking
        As[threadIdx.y][threadIdx.x] = (Arow < width && Acol < width) ? A[Arow * width + Acol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (Brow < width && Bcol < width) ? B[Brow * width + Bcol] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < width && col < width)
        C[row * width + col] = acc;
}

/* Function to initialize matrix with random floats */
void initMatrix(float *mat, int N)
{
    for (int i = 0; i < N * N; ++i)
        mat[i] = (float)(rand()) / RAND_MAX;
}

/* Main function */
int main(int argc, char *argv[])
{
    int N = 1024; // Default matrix size
    if (argc > 1)
        N = atoi(argv[1]);

    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize host matrices
    srand(time(NULL));
    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaCheckError(cudaMalloc((void**)&d_A, bytes));
    cudaCheckError(cudaMalloc((void**)&d_B, bytes));
    cudaCheckError(cudaMalloc((void**)&d_C, bytes));

    // Copy data to device
    cudaCheckError(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1)/TILE_SIZE, (N + TILE_SIZE - 1)/TILE_SIZE);

    // Timing kernel execution
    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&stop));
    cudaCheckError(cudaEventRecord(start));

    matrixMulTiled<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, N);
    cudaCheckError(cudaGetLastError());

    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaEventSynchronize(stop));

    float milliseconds = 0;
    cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result back to host
    cudaCheckError(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Simple verification: print first element
    printf("C[0][0] = %f\n", h_C[0]);

    printf("Kernel execution time: %f ms\n", milliseconds);

    // Clean up
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
