```cu
/*
Implement the operation out-of-place: `Z[i] = a * X[i] + Y[i]`.

Thinking:
The goal is to provide a minimal yet complete CUDA C program that performs the
out‑of‑place element‑wise operation `Z[i] = a * X[i] + Y[i]`.  The program
should:

1. Define a CUDA kernel that computes the formula for each element in parallel.
2. Allocate host (CPU) arrays for X, Y, and Z, and initialize X and Y with
   example values.  The scalar `a` is also defined on the host.
3. Allocate device (GPU) memory for X, Y, and Z, and copy the input data from
   host to device.
4. Launch the kernel with a suitable grid/block configuration that covers all
   `N` elements.  A simple 1D launch is sufficient.
5. Copy the computed result from device back to host.
6. Print a few elements of the result to verify correctness.
7. Clean up all allocated memory.

Error checking is performed after each CUDA API call to aid debugging.
The program is self‑contained, uses only the CUDA runtime API, and can be
compiled with `nvcc` and executed on any CUDA‑enabled GPU.  It demonstrates
the typical workflow of allocating memory, transferring data, launching a
kernel, and retrieving results.  The kernel itself is straightforward and
exposes how to implement an out‑of‑place vector operation in CUDA.
*/

#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel to compute Z[i] = a * X[i] + Y[i]
__global__ void axpy_kernel(const float* X, const float* Y, float* Z, float a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        Z[idx] = a * X[idx] + Y[idx];
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl;                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main() {
    const int N = 10;          // Number of elements
    const float a = 2.0f;      // Scalar multiplier

    // Allocate host memory
    float* h_X = new float[N];
    float* h_Y = new float[N];
    float* h_Z = new float[N];

    // Initialize input arrays
    for (int i = 0; i < N; ++i) {
        h_X[i] = static_cast<float>(i);        // X = [0, 1, 2, ...]
        h_Y[i] = static_cast<float>(N - i);    // Y = [N, N-1, ...]
    }

    // Device pointers
    float *d_X = nullptr, *d_Y = nullptr, *d_Z = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_X, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Z, N * sizeof(float)));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, h_Y, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel: one thread per element
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    axpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_Y, d_Z, a, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_Z, d_Z, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print the results
    std::cout << "Result of Z[i] = a * X[i] + Y[i] (a = " << a << "):" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "Z[" << i << "] = " << h_Z[i] << std::endl;
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
    CUDA_CHECK(cudaFree(d_Z));

    // Free host memory
    delete[] h_X;
    delete[] h_Y;
    delete[] h_Z;

    return 0;
}
```