```cuda
/*********************************************************************
* Aim of the program (exact wording from the prompt):
* 
* Profile and compare the naive vs. tiled matrix multiplication kernels
* using CUDA events. Measure the execution time for a large matrix
* (e.g., 1024x1024) and report the speedup achieved by using shared memory.
* 
* Thinking:
* 1. We need two CUDA kernels: 
*    - naiveMatMul: each thread computes one element of C by iterating
*      over the shared dimension K and accumulating the product.
*    - tiledMatMul: uses shared memory to load tiles of A and B,
*      reducing global memory traffic. The block size is chosen to fit
*      into shared memory and to maximize occupancy (16x16 works for
*      1024x1024 matrices and typical GPUs).
* 
* 2. Host code will:
*    - Allocate host matrices A, B, C_naive, C_tiled.
*    - Initialize A and B with deterministic values for reproducibility.
*    - Allocate device matrices and copy A, B to device.
*    - Create CUDA events for timing.
*    - Launch naive kernel, record time, copy result back.
*    - Launch tiled kernel, record time, copy result back.
*    - Verify that the results are close (optional but useful).
*    - Compute and print the execution times and the speedup factor.
* 
* 3. Timing is performed using cudaEventRecord / cudaEventSynchronize /
*    cudaEventElapsedTime, which gives milliseconds. 
*    We will report the times in milliseconds and the speedup as
*    naive_time / tiled_time.
* 
* 4. For correctness, we use single-precision floats. The matrix
*    dimension is 1024x1024; blockDim is 16x16, so gridDim is 64x64.
* 
* 5. Error checking is performed after each CUDA API call.
*********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

/* --------------------- Error Checking Macro --------------------- */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* --------------------- Kernel: Naive Matrix Multiplication -------- */
__global__ void naiveMatMul(const float* A, const float* B, float* C,
                            int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index of C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // column index of C

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/* --------------------- Kernel: Tiled Matrix Multiplication -------- */
__global__ void tiledMatMul(const float* A, const float* B, float* C,
                            int N)
{
    // Block size (must match launch configuration)
    const int TILE_DIM = 16;

    // Shared memory for tiles of A and B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // Identify thread and block indices
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles of the input matrices
    for (int t = 0; t < (N + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Load tile from A
        if (row < N && t * TILE_DIM + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_DIM + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from B
        if (col < N && t * TILE_DIM + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the two tiles together
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

/* --------------------- Helper: Matrix Initialization ------------- */
void initMatrix(float* mat, int N)
{
    // Simple deterministic initialization for reproducibility
    for (int i = 0; i < N * N; ++i) {
        mat[i] = sinf(i) * cosf(i);
    }
}

/* --------------------- Helper: Verify Result ---------------------- */
bool verifyResult(const float* C1, const float* C2, int N)
{
    const float eps = 1e-3f;
    for (int i = 0; i < N * N; ++i) {
        if (fabsf(C1[i] - C2[i]) > eps) {
            printf("Mismatch at index %d: %f vs %f\n", i, C1[i], C2[i]);
            return false;
        }
    }
    return true;
}

/* --------------------- Main Function ------------------------------ */
int main(void)
{
    const int N = 1024;            // Matrix dimension
    const size_t size = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_naive = (float*)malloc(size);
    float *h_C_tiled = (float*)malloc(size);

    // Initialize host matrices
    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C_naive, *d_C_tiled;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C_naive, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C_tiled, size));

    // Copy input matrices to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* ------------------- Naive Kernel Timing --------------------- */
    CUDA_CHECK(cudaEventRecord(start));
    naiveMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C_naive, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_naive = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_naive, start, stop));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C_naive, size, cudaMemcpyDeviceToHost));

    /* ------------------- Tiled Kernel Timing --------------------- */
    CUDA_CHECK(cudaEventRecord(start));
    tiledMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C_tiled, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_tiled = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_tiled, start, stop));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C_tiled, size, cudaMemcpyDeviceToHost));

    // Verify correctness
    if (!verifyResult(h_C_naive, h_C_tiled, N)) {
        fprintf(stderr, "Verification failed!\n");
    } else {
        printf("Verification passed.\n");
    }

    // Compute speedup
    float speedup = time_naive / time_tiled;

    // Print timings and speedup
    printf("Naive kernel time:   %.3f ms\n", time_naive);
    printf("Tiled kernel time:   %.3f ms\n", time_tiled);
    printf("Speedup (naive / tiled): %.2fx\n", speedup);

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_naive));
    CUDA_CHECK(cudaFree(d_C_tiled));
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_tiled);

    return 0;
}
```