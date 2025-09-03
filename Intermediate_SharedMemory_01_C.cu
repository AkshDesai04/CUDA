```cpp
/*
Time the naive kernel for a 512x512 matrix multiplication.

The goal of this program is to measure the execution time of a simple,
unoptimized (naive) CUDA kernel that performs matrix multiplication
C = A × B where each matrix is of size 512×512. The kernel assigns
one thread per output element; each thread loads the corresponding
row of A and column of B and computes the dot product. We use
CUDA events to capture the GPU execution time of the kernel launch
only, excluding memory transfers. The code includes host-side
initialization of matrices, device memory allocation, copying
to/from device, kernel launch, timing, and a simple correctness
check. The timing result is printed in milliseconds.

Key design choices:
- Block size: 16×16 threads to fit nicely on most GPUs and
  provide a balanced workload.
- Grid size: (512+15)/16 = 32 blocks in each dimension.
- Error checking is performed via a helper macro to keep the
  code clean.
- The kernel uses shared memory only if desired; for a truly
  naive implementation we avoid any optimizations such as tiling.
*/

#include <iostream>
#include <cuda_runtime.h>
#include <cassert>

#define N 512
#define TILE_DIM 16

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in " << __FILE__ << " at line "            \
                      << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

// Naive matrix multiplication kernel: one thread per output element
__global__ void matMulKernel(const float* A, const float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // i
    int col = blockIdx.x * blockDim.x + threadIdx.x; // j

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; ++k) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Host-side naive CPU implementation for verification (optional)
void matMulCPU(const float* A, const float* B, float* C, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < width; ++k) {
                sum += A[i * width + k] * B[k * width + j];
            }
            C[i * width + j] = sum;
        }
    }
}

int main() {
    const int size = N * N;
    const size_t bytes = size * sizeof(float);

    // Allocate host memory
    float* h_A = new float[size];
    float* h_B = new float[size];
    float* h_C = new float[size];
    float* h_C_ref = new float[size]; // For reference

    // Initialize matrices A and B with some values
    for (int i = 0; i < size; ++i) {
        h_A[i] = static_cast<float>(i % 10 + 1);          // Arbitrary values
        h_B[i] = static_cast<float>((i * 3) % 10 + 1);    // Arbitrary values
    }

    // Optional: compute reference result on CPU
    matMulCPU(h_A, h_B, h_C_ref, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch kernel
    matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Compute elapsed time
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Optional: verify correctness
    bool correct = true;
    const float epsilon = 1e-3f;
    for (int i = 0; i < size; ++i) {
        if (abs(h_C[i] - h_C_ref[i]) > epsilon) {
            correct = false;
            std::cerr << "Mismatch at index " << i << ": GPU "
                      << h_C[i] << " vs CPU " << h_C_ref[i] << std::endl;
            break;
        }
    }

    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
    std::cout << "Result correctness: " << (correct ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```