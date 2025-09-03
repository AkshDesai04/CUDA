/*
Aim: Implement naive matrix-vector multiplication, where each thread block computes a portion of the output vector.

Thoughts:
- The goal is to write a simple CUDA program that multiplies a matrix A (size MxN) by a vector x (size N), producing a vector y (size M).
- Each thread block is responsible for computing a chunk of the output vector. Within the block, each thread will compute one element of y (i.e., one row of A).
- Since the multiplication is naive, each thread will iterate over all columns of its assigned row and perform the dot product with the vector x. This is straightforward but may not be the most efficient due to global memory traffic.
- We'll use a 1D grid of 1D blocks. blockDim.x will be chosen (e.g., 256). The total number of blocks will be ceil(M / blockDim.x).
- Memory: We'll allocate host arrays for A, x, y. Copy A and x to device, launch kernel, copy y back, free device memory.
- For simplicity, the program will define small example sizes (e.g., M=6, N=4) and fill with deterministic values to verify correctness.
- We'll include error checking after CUDA calls. The kernel will use row-major storage for A.
- After computation, the host will print the resulting vector to confirm correctness.
- The code will be written in a .cu file, with a multiline comment at the top containing the aim and all reasoning, followed by the implementation.
*/

#include <iostream>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA(call)                                        \
    do {                                                        \
        cudaError_t err = call;                                 \
        if (err != cudaSuccess) {                               \
            std::cerr << "CUDA error at " << __FILE__ << ":"    \
                      << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while (0)

// Kernel: naive matrix-vector multiplication
__global__ void matVecMul(const float *A, const float *x, float *y, int N, int M) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        for (int col = 0; col < N; ++col) {
            sum += A[row * N + col] * x[col];
        }
        y[row] = sum;
    }
}

int main() {
    // Matrix and vector dimensions
    const int M = 6; // number of rows in A, size of output vector y
    const int N = 4; // number of columns in A, size of input vector x

    // Host memory allocation
    float *h_A = new float[M * N];
    float *h_x = new float[N];
    float *h_y = new float[M];

    // Initialize matrix A with deterministic values for testing
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A[i * N + j] = static_cast<float>(i * N + j + 1); // values 1,2,3,...
        }
    }

    // Initialize vector x
    for (int j = 0; j < N; ++j) {
        h_x[j] = static_cast<float>(j + 1); // 1,2,3,...
    }

    // Device memory allocation
    float *d_A, *d_x, *d_y;
    CHECK_CUDA(cudaMalloc((void**)&d_A, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, M * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    matVecMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_x, d_y, N, M);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost));

    // Print the result
    std::cout << "Resulting vector y:" << std::endl;
    for (int i = 0; i < M; ++i) {
        std::cout << h_y[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_x;
    delete[] h_y;
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    return 0;
}
