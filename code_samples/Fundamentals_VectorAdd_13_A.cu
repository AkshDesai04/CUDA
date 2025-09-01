/*
Aim: Implement element-wise minimum.

Thinking:
- The goal is to write a CUDA program that takes two input arrays and produces an output array where each element is the minimum of the corresponding elements from the two input arrays.
- We need a kernel that performs this element-wise operation. Each thread will handle one index of the arrays.
- Host code will allocate memory on both host and device, initialize the input arrays, transfer them to the device, launch the kernel, copy the result back, and finally print the output.
- For simplicity and clarity, we will use a fixed array size (e.g., 12 elements) and use a small, easily readable data set. The kernel launch configuration will be computed based on the array size to cover all elements.
- Error handling will be minimal but sufficient to catch allocation and launch errors.
- We will include all necessary CUDA runtime headers and use the C++ style for ease of use, but the code remains pure CUDA C.
- The final file is a complete .cu file that can be compiled with `nvcc` and run on a system with an NVIDIA GPU.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Kernel to compute element-wise minimum of two arrays
__global__ void elementWiseMin(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fminf(a[idx], b[idx]);
    }
}

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                        \
    do {                                                        \
        cudaError_t err = call;                                 \
        if (err != cudaSuccess) {                               \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorName(err), cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while (0)

int main() {
    const int N = 12;
    const int bytes = N * sizeof(float);

    // Host arrays
    float h_a[N] = {1.0f,  5.0f,  3.0f,  7.0f,  9.0f,  2.0f,
                    4.0f,  8.0f,  6.0f, 10.0f, 11.0f,  0.5f};
    float h_b[N] = {2.0f,  4.0f,  3.5f,  6.5f, 10.0f,  1.5f,
                    5.5f,  7.5f,  5.5f,  9.5f, 12.0f,  0.3f};
    float h_out[N] = {0};

    // Device arrays
    float *d_a, *d_b, *d_out;
    CHECK_CUDA(cudaMalloc((void**)&d_a, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_b, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_out, bytes));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    elementWiseMin<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_out, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Print results
    printf("Element-wise minimum of arrays:\n");
    for (int i = 0; i < N; ++i) {
        printf("min(a[%d] = %f, b[%d] = %f) = %f\n", i, h_a[i], i, h_b[i], h_out[i]);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}
