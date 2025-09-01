```cuda
/*
Implement this for rectangular matrices as well (e.g. C(M,N) = A(M,K) * B(K,N)).

The aim of the program is to perform general matrix multiplication on a GPU using CUDA,
supporting arbitrary rectangular dimensions for the input matrices A (M×K) and B (K×N),
and producing the output matrix C (M×N).  The program will allocate the matrices
on the host, initialize them with deterministic values, transfer them to the device,
run a tiled matrix multiplication kernel that uses shared memory for efficiency,
and finally copy the result back to the host.  For validation, a simple CPU
reference implementation will be used to compare the GPU output.  Performance
timing is measured with CUDA events.  The code is self‑contained in a single .cu
file and can be compiled with nvcc.

The kernel uses a tile size of 16×16 (modifiable via TILE_WIDTH).  Each thread
computes one element of the output matrix.  Shared memory arrays As and Bs
store sub‑tiles of A and B.  Because K may not be a multiple of TILE_WIDTH,
the kernel checks bounds when loading into shared memory and uses zero padding
otherwise.  The loop over the K dimension processes tiles one at a time,
synchronizing threads after each tile load and after the inner multiplication
accumulation.

The host code parses optional command line arguments for the matrix sizes M, N, K
(defaulting to 512×512×512 if none are provided).  Random values are not used
to keep the reference comparison deterministic; instead, A[i][j] = i+j and
B[i][j] = i-j.  The program prints the first few entries of both the GPU and
CPU results for visual confirmation.  Error checking macros are used to catch
CUDA API failures.

All necessary headers are included and the code is fully compilable with:
    nvcc -O2 -o matmul matmul.cu
The program can then be run with optional arguments:
    ./matmul M N K
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define TILE_WIDTH 16

// Error checking macro
#define CHECK_CUDA(call)                                                    \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",     \
                    __FILE__, __LINE__, err, cudaGetErrorString(err),       \
                    #call);                                                 \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

// Kernel for matrix multiplication
__global__ void matMulKernel(const float *A, const float *B, float *C,
                             int M, int N, int K)
{
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // Row and column index of the element to compute
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles of A and B
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < numTiles; ++t)
    {
        // Global indices of the elements to load
        int aRow = row;
        int aCol = t * TILE_WIDTH + threadIdx.x;

        int bRow = t * TILE_WIDTH + threadIdx.y;
        int bCol = col;

        // Load elements into shared memory with bounds checking
        if (aRow < M && aCol < K)
            As[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (bRow < K && bCol < N)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product for this tile
        for (int i = 0; i < TILE_WIDTH; ++i)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    // Write result to global memory
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// CPU reference implementation for validation
void matMulCPU(const float *A, const float *B, float *C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char *argv[])
{
    int M = 512, N = 512, K = 512; // default dimensions
    if (argc == 4)
    {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    printf("Matrix sizes: A(%d,%d) B(%d,%d) C(%d,%d)\n", M, K, K, N, M, N);

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);
    float *h_C_ref = (float *)malloc(sizeC);

    // Initialize host matrices
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
            h_A[i * K + j] = (float)(i + j);

    for (int i = 0; i < K; ++i)
        for (int j = 0; j < N; ++j)
            h_B[i * N + j] = (float)(i - j);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, sizeA));
    CHECK_CUDA(cudaMalloc((void **)&d_B, sizeB));
    CHECK_CUDA(cudaMalloc((void **)&d_C, sizeC));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH,
                 (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // Timing events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    // Launch kernel
    matMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // CPU reference
    printf("Computing CPU reference...\n");
    clock_t cpu_start = clock();
    matMulCPU(h_A, h_B, h_C_ref, M, N, K);
    clock_t cpu_end = clock();
    double cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // Verify results
    int correct = 1;
    for (int i = 0; i < M * N; ++i)
    {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-3)
        {
            correct = 0;
            printf("Mismatch at index %d: GPU=%f CPU=%f\n",
                   i, h_C[i], h_C_ref[i]);
            break;
        }
    }
    if (correct)
        printf("Result verification: PASSED\n");
    else
        printf("Result verification: FAILED\n");

    printf("GPU time: %f ms\n", milliseconds);
    printf("CPU time: %f ms\n", cpu_time);

    // Print first few elements
    printf("First 5 elements of result:\n");
    for (int i = 0; i < 5 && i < M * N; ++i)
        printf("%f ", h_C[i]);
    printf("\n");

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
```