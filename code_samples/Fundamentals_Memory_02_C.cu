```cpp
/* 
Aim of the program (as requested, word for word):

Create a `bool` flag `is_correct` and set it to `false` if any element mismatches. Print a final 'Verification successful' or 'Verification FAILED' message.

Thinking behind the implementation:
- I interpreted the request as a simple CUDA program that performs an element-wise operation on two arrays, then verifies the result on the host.
- The program allocates two input arrays (`a` and `b`) on the host, initializes them with known values, and copies them to the device.
- A CUDA kernel multiplies corresponding elements of `a` and `b` to produce `c` on the device.
- The result is copied back to the host, where the program computes the expected result independently.
- During comparison, a `bool` flag `is_correct` is initialized to `true`. If any element differs beyond a small tolerance, the flag is set to `false`.
- Finally, the program prints "Verification successful" if the flag remains `true`, otherwise it prints "Verification FAILED".
- The code includes basic CUDA error checking and is self-contained, suitable for compiling with `nvcc` into a .cu file. 
*/

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define N (1 << 20)          // 1M elements
#define TOLERANCE 1e-5f

// CUDA kernel for element-wise multiplication
__global__ void multiply(const float *a, const float *b, float *c, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] * b[idx];
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            std::cerr << "CUDA error in " << __FILE__ << ":"        \
                      << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main()
{
    // Allocate host memory
    float *h_a = (float *)malloc(N * sizeof(float));
    float *h_b = (float *)malloc(N * sizeof(float));
    float *h_c = (float *)malloc(N * sizeof(float));
    float *h_c_expected = (float *)malloc(N * sizeof(float));

    if (!h_a || !h_b || !h_c || !h_c_expected) {
        std::cerr << "Failed to allocate host memory." << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i) * 1.0f;
        h_b[i] = static_cast<float>(i) * 2.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_c, N * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    multiply<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute expected result on host
    for (int i = 0; i < N; ++i) {
        h_c_expected[i] = h_a[i] * h_b[i];
    }

    // Verification
    bool is_correct = true;
    for (int i = 0; i < N; ++i) {
        float diff = fabs(h_c[i] - h_c_expected[i]);
        if (diff > TOLERANCE) {
            is_correct = false;
            std::cerr << "Mismatch at index " << i
                      << ": GPU = " << h_c[i]
                      << ", CPU = " << h_c_expected[i]
                      << ", diff = " << diff << std::endl;
            break; // early exit on first mismatch
        }
    }

    if (is_correct) {
        std::cout << "Verification successful" << std::endl;
    } else {
        std::cout << "Verification FAILED" << std::endl;
    }

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_expected);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
```