/*
 * Aim of the program: Write a kernel that performs the operation in-place. It takes a single pointer `a` and computes `a[i] = a[i] + 10;`.
 *
 * Thought process:
 * 1. The kernel must modify the array elements in-place, so the array pointer is passed directly.
 * 2. We need to handle arbitrary array sizes, so each thread will compute its global index and guard against out-of-bounds access.
 * 3. The host code will allocate a sample array, copy it to device memory, launch the kernel, copy the result back, and print the modified values to verify correctness.
 * 4. For simplicity and clarity, error checking macros will be used to catch CUDA API failures.
 * 5. The code is written as a complete CUDA C program that can be compiled with `nvcc` and executed on a compatible device.
 * 6. No external libraries beyond the CUDA runtime and standard C++ headers are required.
 */

#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__     \
                      << " : " << cudaGetErrorString(err) << std::endl;      \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel that adds 10 to each element in-place
__global__ void addTenKernel(int *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += 10;
    }
}

int main() {
    const int N = 16;                      // Size of the array
    size_t size = N * sizeof(int);

    // Host allocation and initialization
    int *h_a = new int[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = i * 2;                    // Example initialization
    }

    // Device allocation
    int *d_a = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addTenKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, N);
    CUDA_CHECK(cudaGetLastError());   // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost));

    // Print results
    std::cout << "Result after adding 10 to each element:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_a[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    delete[] h_a;
    CUDA_CHECK(cudaFree(d_a));

    return 0;
}
