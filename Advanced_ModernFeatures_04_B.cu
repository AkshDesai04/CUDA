```cpp
/*
Use WMMA to multiply matrices of `half` precision numbers, which is the primary use case for Tensor Cores.

[Thinking]
The goal is to implement a matrix multiplication of two half‑precision matrices (float16) by exploiting NVIDIA Tensor Cores via the WMMA API.  WMMA requires the matrices to be tiled into 16×16×16 fragments, which is why the program will operate on sizes that are multiples of 16 (e.g., 128×128).  The program will:

1. Allocate host matrices A (M×K) and B (K×N) in half precision, initialize them with deterministic values, and allocate a host matrix C for the result.
2. Copy A and B to device memory.
3. Launch a CUDA kernel that:
   - Processes one 16×16 tile of the output C per thread block.
   - Uses `wmma::matrix_a` and `wmma::matrix_b` fragments for the tiles of A and B.
   - Uses `wmma::accumulator` fragments (float) for intermediate results.
   - Loops over the K dimension in steps of 16, loading fragments, performing the tensor core multiply-accumulate via `wmma::mma_sync`, and accumulating into a single C fragment.
   - Stores the final C fragment back to the output matrix in row‑major layout.
4. Copy the result back to the host and print a few elements to confirm correctness.
5. Clean up device and host resources.

The kernel uses 256 threads per block, but WMMA internally schedules the operations.  Strides are handled correctly for row‑major matrices.  Error checking is added around CUDA API calls for robustness.  The code compiles with `nvcc -arch=sm_70` or newer.
*/

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Constants for matrix dimensions (must be multiples of 16 for WMMA)
const int M = 128; // Number of rows in A and C
const int N = 128; // Number of columns in B and C
const int K = 128; // Number of columns in A / rows in B

// CUDA error checking macro
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            std::cerr << "CUDA error at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;      \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel to perform matrix multiplication using WMMA
__global__ void wmmaGemm(const __half *A, const __half *B, __half *C)
{
    // Each block computes a 16×16 tile of C
    int tileM = blockIdx.y; // tile row index
    int tileN = blockIdx.x; // tile column index

    // Row and column start indices of the tile in C
    int row = tileM * 16;
    int col = tileN * 16;

    // Declare fragments for the tiles
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Initialize the accumulator fragment to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over the K dimension in 16‑element chunks
    for (int k = 0; k < K / 16; ++k) {
        // Load the A tile
        const __half *a_ptr = A + row * K + k * 16;
        wmma::load_matrix_sync(a_frag, a_ptr, K, wmma::mem_row_major);

        // Load the B tile
        const __half *b_ptr = B + k * 16 * N + col;
        wmma::load_matrix_sync(b_frag, b_ptr, N, wmma::mem_col_major);

        // Multiply and accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store the result tile to C
    __half *c_ptr = C + row * N + col;
    wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
}

int main()
{
    // Allocate host matrices
    std::vector<__half> h_A(M * K);
    std::vector<__half> h_B(K * N);
    std::vector<__half> h_C(M * N, __float2half(0.0f));

    // Initialize A and B with deterministic values
    for (int i = 0; i < M * K; ++i) {
        float val = static_cast<float>(i % 10);
        h_A[i] = __float2half(val);
    }
    for (int i = 0; i < K * N; ++i) {
        float val = static_cast<float>((i + 5) % 10);
        h_B[i] = __float2half(val);
    }

    // Device pointers
    __half *d_A = nullptr;
    __half *d_B = nullptr;
    __half *d_C = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(__half)));

    // Copy input matrices to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice));

    // Launch kernel: grid dims tile 16x16, block dim 256 threads
    dim3 gridDim(N / 16, M / 16);
    dim3 blockDim(256);

    wmmaGemm<<<gridDim, blockDim>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(__half), cudaMemcpyDeviceToHost));

    // Print a few results to verify
    std::cout << "Sample results (C[0][0] to C[3][3]):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float val = __half2float(h_C[i * N + j]);
            std::cout << val << "\t";
        }
        std::cout << std::endl;
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```