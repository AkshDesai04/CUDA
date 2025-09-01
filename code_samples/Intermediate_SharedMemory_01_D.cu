```cpp
/*
Verify the correctness of your naive matrix multiplication against a simple CPU implementation.

Thought Process:
1. **Goal** – The program must compute matrix multiplication twice:
   * On the GPU using a naive kernel where each thread calculates one element by
     iterating over the shared dimension.
   * On the CPU using a straightforward triple‑nested loop.
   The results are compared element‑wise and any discrepancy is reported.

2. **Matrix Size** – We choose a modest size (e.g., 512x512) that comfortably fits
   into typical device memory and can finish quickly on most GPUs.

3. **Memory Layout** – Matrices are stored in row‑major order, i.e.
   element (i,j) is at index `i*N + j`.

4. **Random Initialization** – Fill matrices A and B with pseudo‑random
   floating‑point numbers in the range [0,1) using `rand()`.

5. **GPU Kernel** – Each thread is responsible for a single output element C[i][j].
   The kernel loops over k from 0 to N-1 to accumulate the dot product of
   row i of A and column j of B.

6. **CPU Implementation** – Identical triple‑nested loops as the kernel, but
   running on the host.

7. **Comparison** – After copying the GPU result back to host memory,
   compare each element of the GPU and CPU results.  
   Because of floating‑point arithmetic differences, use a tolerance of
   `1e-5` for the absolute difference.

8. **Error Handling** – Wrap CUDA calls with a macro to check return
   codes and print informative messages on failure.

9. **Reporting** – Print the number of mismatched elements. If zero mismatches
   are found, indicate that the GPU implementation is correct.

10. **Clean‑up** – Free all device and host allocations.

The resulting file is a self‑contained CUDA C program that can be compiled
with `nvcc` and executed on a system with an NVIDIA GPU.
*/

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            std::cerr << "CUDA error in file '" << __FILE__       \
                      << "' in line " << __LINE__ << ": "         \
                      << cudaGetErrorString(err) << std::endl;    \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Naive GPU matrix multiplication kernel
__global__ void matMulKernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Simple CPU matrix multiplication
void matMulCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    const int N = 512; // Matrix dimension (N x N)
    const size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C_gpu = (float*)malloc(bytes);
    float* h_C_cpu = (float*)malloc(bytes);

    if (!h_A || !h_B || !h_C_gpu || !h_C_cpu) {
        std::cerr << "Host memory allocation failed!" << std::endl;
        return EXIT_FAILURE;
    }

    // Seed random generator and initialize matrices A and B
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(std::rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int TILE = 16; // Block dimension
    dim3 blockDim(TILE, TILE);
    dim3 gridDim((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));

    // Compute CPU result
    matMulCPU(h_A, h_B, h_C_cpu, N);

    // Verify correctness
    int mismatches = 0;
    const float epsilon = 1e-5f;
    for (int i = 0; i < N * N; ++i) {
        float diff = std::fabs(h_C_gpu[i] - h_C_cpu[i]);
        if (diff > epsilon) {
            ++mismatches;
            // Uncomment the following line to see first few mismatches
            // if (mismatches <= 10) std::cout << "Mismatch at index " << i << ": GPU=" << h_C_gpu[i] << ", CPU=" << h_C_cpu[i] << "\n";
        }
    }

    if (mismatches == 0) {
        std::cout << "SUCCESS: GPU and CPU results match within tolerance (" << epsilon << ").\n";
    } else {
        std::cout << "FAILURE: Found " << mismatches << " mismatches between GPU and CPU results.\n";
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);

    return EXIT_SUCCESS;
}
```