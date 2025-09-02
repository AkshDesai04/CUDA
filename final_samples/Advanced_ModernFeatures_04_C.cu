```cuda
/*********************************************************************
Aim of the program (word to word):
Profile your WMMA kernel with Nsight Compute and look for the 
`tensor_precision_fu_utilization` metric to confirm Tensor Cores are being used.

Thinking:

1. The goal is to provide a minimal, self-contained CUDA program that
   performs a matrix multiplication using the WMMA API (Tensor Cores)
   so that a user can run Nsight Compute on it and inspect the
   `tensor_precision_fu_utilization` metric.

2. We choose a 2D matrix multiply of dimensions that are multiples of 16
   (the tile size for Tensor Cores).  The classic example is an
   `A (M×K)` multiplied by `B (K×N)` to produce `C (M×N)` where `M, N, K`
   are all multiples of 16.  This guarantees that the WMMA fragment
   sizes are compatible with Tensor Cores.

3. We will use `half` precision for the input matrices and `float`
   precision for the accumulator, which is the most common configuration
   for Tensor Core usage.  The wmma API requires fragments of type
   `wmma::fragment<wmma::matrix_a, 16, 16, 16, wmma::row_major, half>`
   (and similarly for matrix_b).

4. The kernel will load tiles of `A` and `B` from global memory, perform
   the fragment-level multiply-accumulate with `wmma::mma_sync`, and
   write the result back to global memory.  For simplicity we avoid
   shared memory tiling; this still uses Tensor Cores because the WMMA
   API internally uses them regardless of whether shared memory is used.

5. The host code will allocate the matrices, fill them with deterministic
   values (so that correctness can be checked), launch the kernel, and
   copy the result back.  We include a simple reference computation on
   the host to verify correctness.

6. Error checking macros (`cudaCheckError`) are added to catch any CUDA
   API errors.

7. The program compiles with `nvcc -arch=sm_70 -std=c++11 wmma_profile.cu`
   and can be profiled with Nsight Compute using the command:
   `nv-nsight-cu-cli -m tensor_precision_fu_utilization ./wmma_profile`

8. All code is self‑contained and uses only the standard CUDA headers and
   the MMA header (`mma.h`).  No external libraries are required.

*********************************************************************/

#include <cuda.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

using namespace nvcuda;

// Error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Matrix dimensions (must be multiples of 16 for WMMA)
const int M = 256;   // Rows of A and C
const int N = 256;   // Columns of B and C
const int K = 256;   // Columns of A / Rows of B

// Tile size for Tensor Cores
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Kernel: WMMA based matrix multiply
__global__ void wmmaGemmKernel(const half *A, const half *B, float *C)
{
    // Identify the row and column of the C matrix that this thread block
    // should compute.  Each thread block computes a 16x16 tile of C.
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each warp processes one 16x16 tile of C.  Since a warp has 32 threads,
    // each thread computes 1/2 of the tile; we use shared memory for intermediate
    // results to combine them.
    // For simplicity, we let each thread compute a single element of the
    // 16x16 tile using WMMA fragments of size 8x8 (half the tile size).
    // But to fully utilize Tensor Cores, we use 16x16 fragments directly.

    // Declare the WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the output matrix fragment to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over the K dimension in increments of WMMA_K
    for (int tileIdx = 0; tileIdx < K / WMMA_K; ++tileIdx)
    {
        // Offsets for the current tile
        int aRow = blockRow * WMMA_M;
        int aCol = tileIdx * WMMA_K;
        int bRow = tileIdx * WMMA_K;
        int bCol = blockCol * WMMA_N;

        // Load the matrix tiles from global memory to WMMA fragments
        wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
        wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

        // Perform the matrix multiply-accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Write the 16x16 tile of C back to global memory
    int cRow = blockRow * WMMA_M;
    int cCol = blockCol * WMMA_N;
    wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
}

int main()
{
    // Allocate host matrices
    half *h_A = (half*)malloc(M * K * sizeof(half));
    half *h_B = (half*)malloc(K * N * sizeof(half));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    float *h_C_ref = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices A and B with deterministic values
    for (int i = 0; i < M * K; ++i)
    {
        h_A[i] = __float2half(static_cast<float>(i % 100) / 100.0f);
    }
    for (int i = 0; i < K * N; ++i)
    {
        h_B[i] = __float2half(static_cast<float>((i * 2) % 100) / 100.0f);
    }

    // Compute reference result on host (float precision)
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                float a = __half2float(h_A[row * K + k]);
                float b = __half2float(h_B[k * N + col]);
                sum += a * b;
            }
            h_C_ref[row * N + col] = sum;
        }
    }

    // Allocate device matrices
    half *d_A, *d_B;
    float *d_C;
    cudaCheckError(cudaMalloc((void**)&d_A, M * K * sizeof(half)));
    cudaCheckError(cudaMalloc((void**)&d_B, K * N * sizeof(half)));
    cudaCheckError(cudaMalloc((void**)&d_C, M * N * sizeof(float)));

    // Copy data to device
    cudaCheckError(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 threadsPerBlock(32, 1, 1); // one warp per block
    dim3 numBlocks((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M, 1);

    // Launch kernel
    wmmaGemmKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // Copy result back to host
    cudaCheckError(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify correctness
    bool ok = true;
    for (int i = 0; i < M * N; ++i)
    {
        float diff = fabs(h_C[i] - h_C_ref[i]);
        if (diff > 1e-2f)
        {
            printf("Mismatch at index %d: GPU=%f, CPU=%f, diff=%f\n",
                   i, h_C[i], h_C_ref[i], diff);
            ok = false;
            break;
        }
    }
    if (ok)
        printf("Matrix multiplication succeeded and results match.\n");
    else
        printf("Matrix multiplication failed.\n");

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaCheckError(cudaFree(d_A));
    cudaCheckError(cudaFree(d_B));
    cudaCheckError(cudaFree(d_C));

    return 0;
}
```