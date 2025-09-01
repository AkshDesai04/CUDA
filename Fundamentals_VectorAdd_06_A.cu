```cpp
/*
Aim of the program: Implement the operation out-of-place: `Z[i] = a * X[i] + Y[i]`.

Thinking:
- The goal is to provide a CUDA C program that performs a simple element-wise vector operation
  on the GPU.  The operation is out-of-place, meaning that the result is written to a separate
  array Z, leaving the input arrays X and Y untouched.
- We will implement a CUDA kernel `vecAddMul` that takes pointers to X, Y, Z arrays, a scalar
  multiplier `a`, and the number of elements `N`.  Each thread will compute a single element
  of Z: `Z[idx] = a * X[idx] + Y[idx]`.
- For simplicity and clarity, the code will use plain CUDA API calls (`cudaMalloc`,
  `cudaMemcpy`, `cudaLaunchKernel`, etc.) and will include a small host test that
  allocates sample vectors, initializes them, calls the kernel, copies the result back,
  and prints a few elements to verify correctness.
- Basic error checking is provided via the `CUDA_CHECK` macro, which wraps CUDA runtime
  calls and aborts on failure.  This keeps the code robust while remaining concise.
- The vector size `N` is defined as a compile-time constant for demonstration purposes,
  but the kernel can work with any size.  We also calculate the grid and block dimensions
  to cover all elements.
- The program is written entirely in C++ style (with CUDA extensions) and compiles with
  `nvcc`.  It is self-contained and does not rely on external libraries such as Thrust.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << "(" << __LINE__ << "): " \
                      << cudaGetErrorString(err) << std::endl;                \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// CUDA kernel that computes Z[i] = a * X[i] + Y[i] for all i in [0, N)
__global__ void vecAddMul(const float* __restrict__ X,
                          const float* __restrict__ Y,
                          float* __restrict__ Z,
                          float a,
                          size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        Z[idx] = a * X[idx] + Y[idx];
    }
}

int main() {
    const size_t N = 1 << 20; // 1M elements
    const float a = 2.5f;

    // Allocate host memory
    float *h_X = new float[N];
    float *h_Y = new float[N];
    float *h_Z = new float[N];

    // Initialize host data
    for (size_t i = 0; i < N; ++i) {
        h_X[i] = static_cast<float>(i);
        h_Y[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_X, *d_Y, *d_Z;
    CUDA_CHECK(cudaMalloc(&d_X, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Z, N * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, h_Y, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAddMul<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_Y, d_Z, a, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to finish

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_Z, d_Z, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few results
    std::cout << "Sample results (index : X, Y, Z, expected Z):" << std::endl;
    for (size_t i = 0; i < 5; ++i) {
        float expected = a * h_X[i] + h_Y[i];
        std::cout << i << " : " << h_X[i] << ", " << h_Y[i]
                  << " -> " << h_Z[i]
                  << " (expected " << expected << ")" << std::endl;
    }

    // Cleanup
    delete[] h_X;
    delete[] h_Y;
    delete[] h_Z;
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
    CUDA_CHECK(cudaFree(d_Z));

    return 0;
}
```