```cpp
/*
Aim of the program:
For a GPU with Tensor Cores (Volta or newer), use the nvcuda::wmma (Warp-Level Matrix-Multiply-Accumulate) C++ API to perform a small 16x16 matrix multiplication. This involves loading fragments into registers, performing the mma_sync operation, and storing the results. Compare its speed to a standard CUDA core implementation.

Thinking:
1. Check that the device supports Tensor Cores (compute capability >= 7.0). If not, skip wmma part.
2. Create small 16x16 matrices A, B, C on the host. Use half precision for A and B, and float for the accumulator C to keep things simple.
3. Allocate device memory for these matrices. Use cudaMalloc for device arrays.
4. Write two kernels:
   a) wmmaMatMulKernel: uses wmma fragments to perform the matrix multiplication.
      - Load A and B fragments using wmma::load_matrix_sync.
      - Perform mma_sync to accumulate into an accumulator fragment.
      - Store the result back to C using wmma::store_matrix_sync.
   b) cudaMatMulKernel: naive CUDA core implementation. Since the matrices are tiny, we can have one thread per element (or one warp). For simplicity, use a 2D grid of threads.
5. In main, initialize the host matrices with deterministic values (e.g., A[i] = i, B[i] = 2*i).
6. Copy matrices to device, run the wmma kernel, time it using cudaEvent_t. Copy back the result for correctness check.
7. Repeat for the standard CUDA kernel, timing it similarly.
8. Print the time taken by each implementation and optionally verify that the results are close.
9. Use proper error checking after CUDA calls. Ensure the code compiles only for devices with sm_70 or later by using `#if __CUDA_ARCH__ >= 700` around wmma code.
10. Output is a single .cu file containing all of the above.

Edge cases:
- The program should be compilable with nvcc -arch=sm_70.
- If the device does not support Tensor Cores, print a message and skip wmma timing.
- Use cudaDeviceSynchronize after kernel launches to ensure accurate timing.
- Use cudaEventElapsedTime for timing (gives milliseconds).
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// Error checking macro
#define CUDA_CHECK(err)                                                    \
    do {                                                                   \
        cudaError_t err_ = (err);                                          \
        if (err_ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

// Size of the matrices
constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;

// Kernel using WMMA (Tensor Cores)
__global__ void wmmaMatMulKernel(const half *A, const half *B, float *C)
{
    // Each warp will compute one 16x16 tile
    // Since the matrix is 16x16, one warp is enough
    // Load the entire matrices into fragments
    // Use row-major layout
    // fragment<layout, M, N, K, ElemType, AccumType>
    // For 16x16, we can use M=16, N=16, K=16

    // Fragment types
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

    // Initialize output fragment to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load A and B into fragments
    wmma::load_matrix_sync(a_frag, A, M);
    wmma::load_matrix_sync(b_frag, B, M);

    // Perform matrix multiply accumulate
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the result
    wmma::store_matrix_sync(C, c_frag, M, wmma::mem_row_major);
}

// Naive CUDA core implementation
__global__ void cudaMatMulKernel(const half *A, const half *B, float *C)
{
    // Each thread computes one element of the output matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            half a = A[row * K + k];
            half b = B[k * N + col];
            acc += __half2float(a) * __half2float(b);
        }
        C[row * N + col] = acc;
    }
}

int main()
{
    // Check device capability
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s, SM: %d.%d\n", prop.name, prop.major, prop.minor);
    bool wmmaSupported = (prop.major >= 7);
    if (!wmmaSupported)
    {
        printf("Tensor Cores not supported on this device. Exiting.\n");
        return 0;
    }

    // Host matrices
    half *h_A = (half*)malloc(M * K * sizeof(half));
    half *h_B = (half*)malloc(K * N * sizeof(half));
    float *h_C_wmma = (float*)malloc(M * N * sizeof(float));
    float *h_C_cuda = (float*)malloc(M * N * sizeof(float));

    // Initialize A and B with simple values
    for (int i = 0; i < M * K; ++i)
        h_A[i] = __float2half((float)i);
    for (int i = 0; i < K * N; ++i)
        h_B[i] = __float2half((float)(2 * i));

    // Device matrices
    half *d_A, *d_B;
    float *d_C_wmma, *d_C_cuda;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C_wmma, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_cuda, M * N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Launch WMMA kernel
    CUDA_CHECK(cudaEventRecord(start));
    wmmaMatMulKernel<<<1, 32>>>(d_A, d_B, d_C_wmma); // 32 threads (one warp)
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float wmma_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&wmma_ms, start, stop));

    // Copy back result
    CUDA_CHECK(cudaMemcpy(h_C_wmma, d_C_wmma, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Launch CUDA core kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    CUDA_CHECK(cudaEventRecord(start));
    cudaMatMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C_cuda);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float cuda_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&cuda_ms, start, stop));

    // Copy back result
    CUDA_CHECK(cudaMemcpy(h_C_cuda, d_C_cuda, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify correctness (allow small tolerance)
    bool correct = true;
    for (int i = 0; i < M * N; ++i)
    {
        float diff = fabs(h_C_wmma[i] - h_C_cuda[i]);
        if (diff > 1e-3f)
        {
            correct = false;
            printf("Mismatch at %d: WMMA=%f, CUDA=%f\n", i, h_C_wmma[i], h_C_cuda[i]);
            break;
        }
    }
    printf("Results %s\n", correct ? "match" : "do NOT match");

    // Print timings
    printf("WMMA (Tensor Cores) time: %f ms\n", wmma_ms);
    printf("CUDA core time: %f ms\n", cuda_ms);

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_wmma));
    CUDA_CHECK(cudaFree(d_C_cuda));
    free(h_A);
    free(h_B);
    free(h_C_wmma);
    free(h_C_cuda);

    return 0;
}
```