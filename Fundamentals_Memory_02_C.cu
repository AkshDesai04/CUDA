/*
Aim of the program:
Create a `bool` flag `is_correct` and set it to `false` if any element mismatches. Print a final 'Verification successful' or 'Verification FAILED' message.

Thinking:
To demonstrate the use of a `bool` flag for verification we perform a simple vector addition on both the CPU and GPU.  
1. Allocate two input arrays `h_A` and `h_B` and fill them with deterministic values.  
2. Allocate an output array `h_C_cpu` for the CPU result and compute the addition in a simple loop.  
3. Allocate device memory for `d_A`, `d_B`, `d_C` and copy the input data to the GPU.  
4. Launch a CUDA kernel that performs element‑wise addition and writes the result to `d_C`.  
5. Copy `d_C` back to the host into `h_C_gpu`.  
6. Compare `h_C_cpu` and `h_C_gpu` element by element.  
   - If any pair differs beyond a small epsilon, set `is_correct` to `false` and break.  
7. Finally print either "Verification successful" (if `is_correct` remains true) or "Verification FAILED" otherwise.  

The program uses standard error checking macros and ensures all allocated memory is freed before exit. The kernel launch configuration is chosen based on the array size to maximize occupancy. This example is self‑contained and can be compiled with `nvcc -o vector_add vector_add.cu` and executed on any machine with CUDA support.
*/

#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Macro for checking CUDA errors following a CUDA launch or API call
#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__   \
                      << " - " << cudaGetErrorString(err) << std::endl;     \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1 << 20;  // 1 million elements
    const size_t size = N * sizeof(float);

    // Host allocations
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C_cpu = new float[N];
    float *h_C_gpu = new float[N];

    // Initialize input arrays with deterministic values
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    // CPU computation
    for (int i = 0; i < N; ++i) {
        h_C_cpu[i] = h_A[i] + h_B[i];
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));

    // Verification
    bool is_correct = true;
    const float epsilon = 1e-5f;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C_cpu[i] - h_C_gpu[i]) > epsilon) {
            is_correct = false;
            std::cerr << "Mismatch at index " << i
                      << ": CPU=" << h_C_cpu[i]
                      << ", GPU=" << h_C_gpu[i] << std::endl;
            break;
        }
    }

    if (is_correct) {
        std::cout << "Verification successful" << std::endl;
    } else {
        std::cout << "Verification FAILED" << std::endl;
    }

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
