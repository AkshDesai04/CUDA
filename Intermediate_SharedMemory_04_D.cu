/*
Compare your tiled kernel's performance against the `cublasSgemm` function from the cuBLAS library. (cuBLAS will be much faster).

Thinking Process:
1. The goal is to create a CUDA program that performs matrix multiplication using a tiled shared‑memory kernel and then compares its performance against the highly optimised cuBLAS routine `cublasSgemm`. 
2. I decided to use single‑precision floating point matrices (float) because cuBLAS' `cublasSgemm` operates on `float`.  
3. For a fair comparison, both kernels must work on the same data. cuBLAS expects matrices in column‑major order, so I store all matrices in that layout.  
4. The tiled kernel will be written to load sub‑tiles of the input matrices into shared memory, multiply them, and accumulate into the output tile.  
5. Timing will be done with CUDA events (`cudaEvent_t`) to capture device execution time.  
6. I will generate random matrices on the host, copy them to device, run the tiled kernel, then run cuBLAS, and finally copy back both results to the host.  
7. To verify correctness I will compute the maximum absolute difference between the two results and print it.  
8. For simplicity the tile size is set to 32 (typical warp size), but it can be adjusted.  
9. Error checking macros are used for CUDA and cuBLAS API calls.  
10. The program accepts an optional command‑line argument to set the matrix dimension (NxN). If omitted, it defaults to 1024.  
11. All necessary headers are included (`<cstdio>`, `<cstdlib>`, `<cuda_runtime.h>`, `<cublas_v2.h>`).  
12. After kernel launch, we synchronize and record times.  
13. Results (timings, GFLOPS, max error) are printed to the console.  
14. The code is self‑contained and can be compiled with `nvcc matrix_compare.cu -lcublas -o matrix_compare`.  
15. The program will report cuBLAS as faster, demonstrating the efficiency of the library.  
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE_SIZE 32
#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

#define CHECK_CUBLAS(call)                                         \
    do {                                                           \
        cublasStatus_t status = (call);                            \
        if (status != CUBLAS_STATUS_SUCCESS) {                    \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n",        \
                    __FILE__, __LINE__, status);                  \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

// Utility function to generate random float matrix in column-major order
void init_matrix(float *mat, int N) {
    for (int i = 0; i < N * N; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Kernel: tiled matrix multiplication C = A * B (column-major)
__global__ void tiled_gemm(const float *A, const float *B, float *C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float value = 0.0f;

    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        // Load subtiles into shared memory
        int tiledRow = row;
        int tiledCol = m * TILE_SIZE + threadIdx.x;
        if (tiledRow < N && tiledCol < N)
            As[threadIdx.y][threadIdx.x] = A[tiledRow + tiledCol * N];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        tiledRow = m * TILE_SIZE + threadIdx.y;
        tiledCol = col;
        if (tiledRow < N && tiledCol < N)
            Bs[threadIdx.y][threadIdx.x] = B[tiledRow + tiledCol * N];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the two tiles together
        for (int e = 0; e < TILE_SIZE; ++e)
            value += As[threadIdx.y][e] * Bs[e][threadIdx.x];

        __syncthreads();
    }

    // Write result to global memory
    if (row < N && col < N)
        C[row + col * N] = value;
}

int main(int argc, char **argv) {
    int N = 1024; // Default matrix size
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "Invalid matrix size.\n");
            return EXIT_FAILURE;
        }
    }

    printf("Matrix size: %d x %d\n", N, N);

    size_t size = N * N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_kernel = (float*)malloc(size);
    float *h_C_cublas = (float*)malloc(size);

    // Initialize host matrices
    srand(1234);
    init_matrix(h_A, N);
    init_matrix(h_B, N);

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size));

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch tiled kernel
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE,
                 (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    tiled_gemm<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    double gflops = (2.0 * N * N * N) / (milliseconds * 1e6); // GFLOPs
    printf("Tiled kernel time: %.3f ms, GFLOPs: %.2f\n", milliseconds, gflops);

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_C_kernel, d_C, size, cudaMemcpyDeviceToHost));

    // cuBLAS routine
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.0f, beta = 0.0f;

    // cuBLAS uses column-major; our data is already column-major
    // cublasSgemm(A, B, C) where C = alpha*A*B + beta*C
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha,
                             d_A, N,
                             d_B, N,
                             &beta,
                             d_C, N));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaMemcpy(h_C_cublas, d_C, size, cudaMemcpyDeviceToHost));

    milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    gflops = (2.0 * N * N * N) / (milliseconds * 1e6);
    printf("cuBLAS time: %.3f ms, GFLOPs: %.2f\n", milliseconds, gflops);

    // Verify correctness
    float max_err = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        float diff = fabs(h_C_kernel[i] - h_C_cublas[i]);
        if (diff > max_err) max_err = diff;
    }
    printf("Max error between tiled kernel and cuBLAS: %e\n", max_err);

    // Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_kernel);
    free(h_C_cublas);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
