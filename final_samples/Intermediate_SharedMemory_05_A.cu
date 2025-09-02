/*
Add boundary checks to the tile loading phase. `if (row < N && col < N) { a_tile[...] = A[...]; } else { a_tile[...] = 0.0f; }`.
I will implement a tiled matrix multiplication kernel that includes boundary checks when loading tiles. This ensures correct handling for non‑multiple‑of‑block‑size matrices. The rest of the program includes memory allocation, data initialization, kernel launch, and result verification. The tile size is chosen as a compile‑time constant (TILE_SIZE) and the kernel will correctly handle matrices of any size N by checking bounds before reading from global memory and writing to shared memory. The host code will also check for CUDA errors and free resources appropriately.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16  // You can change this to 32 for better performance on modern GPUs

// CUDA error checking macro
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel for tiled matrix multiplication with boundary checks during tile load
__global__ void matMulTile(const float *A, const float *B, float *C, int N)
{
    // Calculate global row and column index
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    // Loop over tiles of A and B
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t)
    {
        __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
        __shared__ float b_tile[TILE_SIZE][TILE_SIZE];

        // Global indices for the current tile
        int a_row = row;
        int a_col = t * TILE_SIZE + threadIdx.x;

        int b_row = t * TILE_SIZE + threadIdx.y;
        int b_col = col;

        // Load A tile with boundary checks
        if (a_row < N && a_col < N)
            a_tile[threadIdx.y][threadIdx.x] = A[a_row * N + a_col];
        else
            a_tile[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile with boundary checks
        if (b_row < N && b_col < N)
            b_tile[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        else
            b_tile[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; ++k)
            value += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];

        __syncthreads();
    }

    // Write the result with boundary check
    if (row < N && col < N)
        C[row * N + col] = value;
}

// Helper function to fill matrix with random data
void randomFill(float *data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = (float)(rand() % 100) / 10.0f;
}

// Helper function to print matrix
void printMatrix(const float *data, int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
            printf("%6.2f ", data[i * N + j]);
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    int N = 512; // Default matrix size
    if (argc > 1)
        N = atoi(argv[1]);

    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_C_ref = (float *)malloc(size);

    if (!h_A || !h_B || !h_C || !h_C_ref)
    {
        fprintf(stderr, "Failed to allocate host matrices.\n");
        return EXIT_FAILURE;
    }

    // Initialize host matrices
    srand(0);
    randomFill(h_A, N * N);
    randomFill(h_B, N * N);

    // Reference calculation (CPU) for verification
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += h_A[i * N + k] * h_B[k * N + j];
            h_C_ref[i * N + j] = sum;
        }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matMulTile<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify result
    bool correct = true;
    const float epsilon = 1e-3f;
    for (int i = 0; i < N * N; ++i)
    {
        if (fabs(h_C[i] - h_C_ref[i]) > epsilon)
        {
            correct = false;
            printf("Mismatch at index %d: GPU=%f, CPU=%f\n", i, h_C[i], h_C_ref[i]);
            break;
        }
    }

    if (correct)
        printf("Matrix multiplication successful, results match.\n");
    else
        printf("Matrix multiplication failed.\n");

    // Optional: print matrices for small N
    if (N <= 8)
    {
        printf("Matrix A:\n");
        printMatrix(h_A, N);
        printf("Matrix B:\n");
        printMatrix(h_B, N);
        printf("Result C (GPU):\n");
        printMatrix(h_C, N);
        printf("Reference C (CPU):\n");
        printMatrix(h_C_ref, N);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
