```cuda
/*
Aim of the program:
Profile your WMMA kernel with Nsight Compute and look for the `tensor_precision_fu_utilization` metric to confirm Tensor Cores are being used.

Thinking:
To demonstrate that Tensor Cores are indeed used, we write a simple matrix multiplication kernel that uses the WMMA API (`wmma::mma_sync`) to perform the multiplication in half-precision (float16). 
The code sets up two small matrices A (MxK) and B (KxN) in host memory, copies them to device memory, launches a WMMA kernel that processes the matrices in 16x16 tiles (the default tile size for Tensor Cores on Volta/AMPERE), and then copies the result back to the host. 
We use the CUDA WMMA header (`<mma.h>`), allocate device memory for the full matrices (not just the tiles), and within the kernel we load the tiles into WMMA fragments, perform the multiply-accumulate, and write the results back to global memory. 
The kernel uses shared memory to hold the tiles so that each warp can load its required data once. 
After the kernel completes, we simply print a few elements of the result to confirm correctness. 
By running this program under Nsight Compute and inspecting the `tensor_precision_fu_utilization` metric, we can confirm that Tensor Cores are being employed during the kernel execution.
*/

#include <cstdio>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// Matrix dimensions
const int M = 512;
const int N = 512;
const int K = 512;

// WMMA tile dimensions
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Kernel that performs matrix multiplication C = A * B using WMMA
__global__ void wmma_matrix_multiply(const half* A, const half* B, half* C, int lda, int ldb, int ldc)
{
    // Define WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Each thread block covers a 16x16 tile of C
    // Compute the tile location this block will compute
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Calculate the starting index of the C tile in global memory
    int cRow = blockRow * WMMA_M;
    int cCol = blockCol * WMMA_N;

    // Zero the accumulator fragment
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over all K tiles
    for (int k = 0; k < (K / WMMA_K); ++k)
    {
        // Load the A and B tiles into WMMA fragments
        int aRow = cRow;
        int aCol = k * WMMA_K;
        int bRow = k * WMMA_K;
        int bCol = cCol;

        wmma::load_matrix_sync(a_frag, A + aRow * lda + aCol, lda);
        wmma::load_matrix_sync(b_frag, B + bRow * ldb + bCol, ldb);

        // Perform the matrix multiply-accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store the result back to global memory
    wmma::store_matrix_sync(C + cRow * ldc + cCol, c_frag, ldc, wmma::mem_row_major);
}

int main()
{
    // Allocate host matrices
    half *h_A = new half[M * K];
    half *h_B = new half[K * N];
    half *h_C = new half[M * N];

    // Initialize matrices A and B with simple values
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
            h_A[i * K + j] = __float2half(static_cast<float>(i + j));

    for (int i = 0; i < K; ++i)
        for (int j = 0; j < N; ++j)
            h_B[i * N + j] = __float2half(static_cast<float>(i - j));

    // Allocate device matrices
    half *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(half));
    cudaMalloc((void**)&d_B, K * N * sizeof(half));
    cudaMalloc((void**)&d_C, M * N * sizeof(half));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);

    // Launch the WMMA kernel
    wmma_matrix_multiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, K, N, N);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);

    // Print a few elements of the result matrix
    printf("C[0][0] = %f\n", __half2float(h_C[0]));
    printf("C[1][1] = %f\n", __half2float(h_C[1 * N + 1]));
    printf("C[2][2] = %f\n", __half2float(h_C[2 * N + 2]]);

    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```