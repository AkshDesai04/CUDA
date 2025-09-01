/*
Aim: Modify the kernel to take a fourth argument, `int scalar`, and compute `b[i] = a[i] + scalar;`.

Thinking:
- The task is to update an existing CUDA kernel so that it accepts an additional integer argument (`scalar`) and uses it to add to each element of array `a` when storing into array `b`.
- The kernel will be simple: it will compute the global index `i` for each thread, check bounds, and then perform the addition.
- On the host side, we need to allocate input and output arrays on both host and device, copy the input to the device, launch the kernel with the scalar argument, copy the result back, and finally verify correctness.
- We'll include basic error checking for CUDA API calls and print some sample outputs to confirm the operation.
- The program will use 1D arrays with a size of 1024 for demonstration, but it can be adapted to larger sizes easily.
- CUDA launch parameters: block size of 256 threads, grid size computed to cover all elements.
- The code is written in a single `.cu` file, includes all necessary headers, and can be compiled with `nvcc`.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// Error checking macro
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " - " << cudaGetErrorString(err) << std::endl;      \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

// Kernel that adds a scalar to each element of array a and writes to array b
__global__ void addScalar(const int *a, int *b, int N, int scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        b[idx] = a[idx] + scalar;
    }
}

int main() {
    const int N = 1024;           // Number of elements
    const int scalar = 10;        // Scalar to add

    // Host memory allocation
    int *h_a = static_cast<int*>(malloc(N * sizeof(int)));
    int *h_b = static_cast<int*>(malloc(N * sizeof(int)));
    if (!h_a || !h_b) {
        std::cerr << "Failed to allocate host memory." << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize host input array
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
    }

    // Device memory allocation
    int *d_a = nullptr;
    int *d_b = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(int)));

    // Copy input from host to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addScalar<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N, scalar);

    // Check for launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_b, d_b, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Simple verification: print first 10 results
    std::cout << "First 10 results (a[i] + scalar):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "h_a[" << i << "] = " << h_a[i]
                  << ", h_b[" << i << "] = " << h_b[i] << std::endl;
    }

    // Cleanup
    free(h_a);
    free(h_b);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));

    return 0;
}
