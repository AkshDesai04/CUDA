/*
Aim of the program: A kernel that computes `C[i] = a*A[i] + b*B[i]` is a fused operation.

Thinking:
- The goal is to implement a CUDA kernel that performs a fused multiply-add (FMA) operation on two input arrays A and B, producing output array C. The operation is C[i] = a*A[i] + b*B[i].
- We'll write a simple host program that allocates memory on the host, initializes two input arrays with some sample values, copies them to the GPU, launches the kernel, copies the result back, and finally verifies the result on the host.
- For simplicity and clarity, the code uses single-precision floats. It also includes basic error checking via a macro to catch CUDA API errors.
- The kernel will be launched with a 1D grid of blocks, each block containing 256 threads (typical block size). The grid size is computed to cover all elements of the arrays.
- The program is self-contained: it includes all necessary headers, defines the kernel, host memory allocation and copying, kernel launch, and result validation.
- The fused operation can potentially be optimized by the compiler to use the hardware FMA instruction if the architecture supports it; however, we simply compute the expression as written.
- The code is written in a single .cu file, ready to be compiled with nvcc. No external dependencies other than CUDA toolkit are required.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call or kernel launch.
#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",             \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// Kernel performing the fused operation: C[i] = a*A[i] + b*B[i]
__global__ void fusedAddMul(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            const float a,
                            const float b,
                            size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = a * A[idx] + b * B[idx];
    }
}

int main() {
    const size_t N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i) * 0.5f;
        h_B[i] = static_cast<float>(i) * 1.5f;
    }

    const float a = 2.0f;
    const float b = 3.0f;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    fusedAddMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, a, b, N);
    CHECK_CUDA(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify results
    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        float expected = a * h_A[i] + b * h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %zu: host %f, device %f, expected %f\n",
                    i, h_C[i], expected, expected);
            ok = false;
            break;
        }
    }
    if (ok) {
        printf("All results are correct.\n");
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return EXIT_SUCCESS;
}
