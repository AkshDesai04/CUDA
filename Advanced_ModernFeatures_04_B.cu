/*
Use WMMA to multiply matrices of `half` precision numbers, which is the primary use case for Tensor Cores.

Thought process:

- We need to write a CUDA program that uses the WMMA API to perform matrix multiplication on half-precision matrices.
- The program will allocate two input matrices A (M×K) and B (K×N) in half precision, and compute C = A × B.
- The WMMA API requires the matrices to be laid out in row‑major/column‑major format; we’ll store A in row‑major, B in row‑major but load it as col‑major, and C in row‑major.
- Each CUDA block will compute a 16×16 tile of the result matrix C. The WMMA fragment size of 16×16×16 is used, matching the Tensor Core tile size.
- The kernel uses wmma::fragment<wmma::matrix_a>, <matrix_b>, and <accumulator> types, loads tiles with wmma::load_matrix_sync, multiplies with wmma::mma_sync, and stores the result with wmma::store_matrix_sync.
- The host code initializes the matrices with simple values (e.g., all ones) so that the expected result is easy to verify: each element of C should equal K.
- Memory allocation, copying, kernel launch, synchronization, and result verification are all handled with CUDA runtime API calls and basic error checking.
- We keep the code minimal and self‑contained, using only standard CUDA headers and the WMMA API. The target architecture is sm_70+ to enable Tensor Cores.
*/

#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            std::cerr << "CUDA error in " << __FILE__ << ":"        \
                      << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Matrix dimensions (must be multiples of 16 for WMMA)
const int M = 128;  // Rows of A and C
const int N = 128;  // Columns of B and C
const int K = 128;  // Columns of A / Rows of B

// WMMA kernel: each block computes a 16x16 tile of C
__global__ void wmmaMatrixMulHalf(const half *A, const half *B, half *C, int M, int N, int K)
{
    // Block indices
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Starting row and column of the C tile
    int row = blockRow * 16;
    int col = blockCol * 16;

    // Declare WMMA fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    // Initialize the output accumulator to zero
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over K dimension in steps of 16
    for (int k = 0; k < K; k += 16)
    {
        // Load 16x16 tile of A from global memory
        // Leading dimension is K (row-major)
        wmma::load_matrix_sync(a_frag, A + row * K + k, K);

        // Load 16x16 tile of B from global memory
        // B is stored row-major but loaded as col-major; leading dimension is N
        wmma::load_matrix_sync(b_frag, B + k * N + col, N);

        // Perform the matrix multiplication: acc = a * b + acc
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Store the accumulated result back to global memory
    // C is row-major; leading dimension is N
    wmma::store_matrix_sync(C + row * N + col, acc_frag, N, wmma::mem_row_major);
}

int main()
{
    // Allocate host memory
    size_t sizeA = M * K * sizeof(half);
    size_t sizeB = K * N * sizeof(half);
    size_t sizeC = M * N * sizeof(half);

    half *h_A = new half[M * K];
    half *h_B = new half[K * N];
    half *h_C = new half[M * N];

    // Initialize A and B with 1.0f (converted to half)
    for (int i = 0; i < M * K; ++i)
        h_A[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; ++i)
        h_B[i] = __float2half(1.0f);

    // Allocate device memory
    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeC));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, sizeC));

    // Define grid and block dimensions
    dim3 blockDim(256, 1, 1);                          // 256 threads per block (required for WMMA)
    dim3 gridDim((N + 15) / 16, (M + 15) / 16, 1);     // One block per 16x16 tile

    // Launch kernel
    wmmaMatrixMulHalf<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Verify a few elements
    bool correct = true;
    float expected = static_cast<float>(K);  // Since each row of A and column of B are all ones
    for (int i = 0; i < M && correct; ++i)
    {
        for (int j = 0; j < N && correct; ++j)
        {
            float val = __half2float(h_C[i * N + j]);
            if (fabs(val - expected) > 1e-3f)
            {
                std::cerr << "Mismatch at (" << i << "," << j << "): "
                          << val << " != " << expected << std::endl;
                correct = false;
            }
        }
    }

    if (correct)
        std::cout << "Matrix multiplication succeeded. All values match expected result." << std::endl;
    else
        std::cout << "Matrix multiplication failed." << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
