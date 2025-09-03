```cpp
/*
Aim of the program: 
"Can you still get out-of-bounds errors with a grid-stride loop if `N` is not a multiple of anything? (No, the loop condition `i < N` handles it perfectly)."

Thinking:
The user’s question is about safety of grid‑stride loops when the array size N is not a multiple of the total number of threads. The loop condition `i < N` guarantees that any thread will exit before it accesses an element beyond the array bounds. The goal of this program is to demonstrate this fact in a concrete CUDA example. 

I will write a simple kernel that increments each element of an array by 1. The kernel uses a classic grid‑stride loop, and the loop condition ensures that even if `N` is not divisible by the total thread count, no out‑of‑bounds memory access occurs. The host code will allocate an array, initialize it, launch the kernel, copy the result back, and verify that all elements were incremented correctly. I will also include a small error‑checking macro to illustrate typical CUDA error handling. The program will be fully self‑contained in a single .cu file and can be compiled with `nvcc`. 
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

// Simple CUDA error checking macro
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__    \
                      << " - " << cudaGetErrorString(err) << std::endl;     \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that increments each element by 1 using a grid‑stride loop
__global__ void incrementKernel(int *data, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < N; i += stride) {
        // No out-of-bounds access because of the loop condition i < N
        data[i] += 1;
    }
}

int main() {
    const size_t N = 1 << 20; // Example: 1M elements (not a multiple of typical block counts)
    const int initValue = 5;  // initial value for all elements

    // Allocate host memory
    int *h_data = (int *)malloc(N * sizeof(int));
    if (!h_data) {
        std::cerr << "Failed to allocate host memory." << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize host array
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = initValue;
    }

    // Allocate device memory
    int *d_data;
    CHECK_CUDA(cudaMalloc((void **)&d_data, N * sizeof(int)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel: choose 256 threads per block, 64 blocks (typical)
    const int threadsPerBlock = 256;
    const int blocksPerGrid = 64; // total threads = 16384, N=1M => not a multiple
    incrementKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    bool success = true;
    for (size_t i = 0; i < N; ++i) {
        if (h_data[i] != initValue + 1) {
            std::cerr << "Mismatch at index " << i
                      << ": expected " << initValue + 1
                      << ", got " << h_data[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "All elements incremented correctly. No out-of-bounds errors occurred." << std::endl;
    }

    // Clean up
    free(h_data);
    CHECK_CUDA(cudaFree(d_data));

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```