/*
Allocate and copy a much larger array, e.g., 1 million integers.

Thinking:
The goal is to demonstrate allocation of a large array on both host and device, copying data between them, and optionally performing a simple kernel operation on the device. 1 million integers occupy about 4 MB, which is trivial for modern GPUs, but it serves as a realistic example of allocating a non‑trivial amount of memory.

Key steps:
1. Allocate a host array of size N = 1 000 000.
2. Initialize it with some values (e.g., 0…N-1).
3. Allocate device memory with cudaMalloc.
4. Copy the host array to the device with cudaMemcpy.
5. Launch a simple kernel that, for instance, adds 1 to every element.
6. Copy the result back to the host.
7. Verify a few elements and clean up.

To make the program robust, we include a small error‑checking helper that aborts on CUDA errors.  We also print a few elements before and after the kernel launch to confirm that the data moved correctly and the kernel performed as expected.

The code is a complete, self‑contained CUDA program that can be compiled with nvcc and executed on any CUDA‑capable device.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

// Error checking macro
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            std::cerr << "CUDA error in file '" << __FILE__        \
                      << "' in line " << __LINE__ << ": "          \
                      << cudaGetErrorString(err) << std::endl;      \
            std::exit(EXIT_FAILURE);                                \
        }                                                           \
    } while (0)

// Simple kernel that adds 1 to each element
__global__ void add_one(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] += 1;
    }
}

int main() {
    const int N = 1'000'000;          // 1 million integers
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Host allocation and initialization
    std::vector<int> h_vec(N);
    for (int i = 0; i < N; ++i) {
        h_vec[i] = i;
    }

    // Print first 5 elements before copy
    std::cout << "Host before copy (first 5): ";
    for (int i = 0; i < 5; ++i) {
        std::cout << h_vec[i] << " ";
    }
    std::cout << std::endl;

    // Device allocation
    int *d_vec = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vec, N * sizeof(int)));

    // Copy from host to device
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    add_one<<<blocksPerGrid, threadsPerBlock>>>(d_vec, N);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

    // Copy back from device to host
    CUDA_CHECK(cudaMemcpy(h_vec.data(), d_vec, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print first 5 elements after kernel execution
    std::cout << "Host after kernel (first 5): ";
    for (int i = 0; i < 5; ++i) {
        std::cout << h_vec[i] << " ";
    }
    std::cout << std::endl;

    // Verify correctness for a few elements
    bool success = true;
    for (int i = 0; i < 10; ++i) {
        if (h_vec[i] != i + 1) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": expected "
                      << i + 1 << ", got " << h_vec[i] << std::endl;
        }
    }
    if (success) {
        std::cout << "Verification succeeded for first 10 elements." << std::endl;
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_vec));

    return 0;
}
