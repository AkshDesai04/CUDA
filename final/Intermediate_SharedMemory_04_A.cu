```cpp
/*
Aim of the program (word for word):
Calculate and report the performance in GFLOPS (billions of floating-point operations per second).
The number of FLOPS for matrix multiplication is `2 * N^3`.

Thinking:
- I need a CUDA program that multiplies two N×N matrices and measures how many floating‑point operations
  per second are achieved on the GPU.
- The floating‑point operations count for matrix multiplication is given by 2·N³.
- To obtain the GFLOPS value I will:
    1. Allocate and initialize two input matrices (A and B) and an output matrix (C) on the host.
    2. Copy A and B to the device.
    3. Launch a matrix‑multiplication kernel on the GPU.
    4. Measure the execution time of the kernel using CUDA events.
    5. Compute the number of FLOPS as 2·N³ and divide by the elapsed time (converted to seconds).
    6. Convert to gigaflops (divide by 1e9) and print the result.
- For simplicity and good performance I’ll use a tiled kernel that loads sub‑blocks of A and B into shared
  memory. This is a standard approach that gives decent performance for many sizes.
- The program will accept the matrix size N as a command line argument; if omitted it will default to
  1024. This makes it easy to test the performance on different sizes.
- I’ll use a simple error‑checking macro to keep the code readable.
- All code is written in C/C++ with CUDA extensions, compiled as a single .cu file.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                   \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* Tile size for shared memory */
#define TILE_SIZE 32

/* Kernel for matrix multiplication using shared memory tiling */
__global__ void matMulKernel(const float *A, const float *B, float *C, int N)
{
    /* Shared memory for tiles of A and B */
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float value = 0.0f;

    /* Loop over tiles */
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        /* Load tile of A */
        int aRow = row;
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (aRow < N && aCol < N)
            As[threadIdx.y][threadIdx.x] = A[aRow * N + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        /* Load tile of B */
        int bRow = t * TILE_SIZE + threadIdx.y;
        int bCol = col;
        if (bRow < N && bCol < N)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        /* Multiply the two tiles */
        for (int k = 0; k < TILE_SIZE; ++k)
            value += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    /* Write result */
    if (row < N && col < N)
        C[row * N + col] = value;
}

/* Simple function to initialize a matrix with random floats */
void initMatrix(float *mat, int N)
{
    for (int i = 0; i < N * N; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main(int argc, char *argv[])
{
    /* Parse matrix size from command line */
    int N = 1024; /* Default size */
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "Invalid matrix size.\n");
            return EXIT_FAILURE;
        }
    }

    size_t size = N * N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed.\n");
        return EXIT_FAILURE;
    }

    /* Initialize matrices A and B */
    srand((unsigned)time(NULL));
    initMatrix(h_A, N);
    initMatrix(h_B, N);

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    /* Copy host data to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Define grid and block dimensions */
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE,
                 (N + TILE_SIZE - 1) / TILE_SIZE);

    /* Create CUDA events for timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* Record start event */
    CUDA_CHECK(cudaEventRecord(start, 0));

    /* Launch kernel */
    matMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    /* Record stop event */
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    /* Compute elapsed time in milliseconds */
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    /* Compute GFLOPS */
    double ops = 2.0 * N * N * N;            // 2 * N^3
    double gflops = ops / (milliseconds / 1000.0) / 1e9;

    /* Copy result back to host (optional) */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Print results */
    printf("Matrix size: %d x %d\n", N, N);
    printf("Kernel execution time: %f ms\n", milliseconds);
    printf("GFLOPS: %f\n", gflops);

    /* Clean up */
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
```