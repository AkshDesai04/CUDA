/*
Aim: Allocate an array of N structs on the device. Launch a kernel with N threads. Each thread i should modify the struct at d_data[i].

Thinking: 
The requirement is to demonstrate CUDA memory allocation for a struct type, a simple kernel that writes to each element of the array, and a host program that copies the data back and verifies it. 
1. Define a simple C‑style struct `Data` that contains a couple of fields so that it is non‑trivial to copy.
2. Allocate an array of `N` such structs on the device with `cudaMalloc`.
3. Write a kernel `modify_struct` that takes a pointer to `Data` and writes distinct values to each element based on the thread index.
4. Launch the kernel with `N` threads, using a single block (or multiple blocks if `N` is large). The kernel will be trivial: each thread writes to `d_data[threadIdx.x]`.
5. After execution, copy the array back to the host with `cudaMemcpy`.
6. Print a few elements to verify that each thread updated its corresponding struct.
7. Add simple error checking after each CUDA call for robustness.
8. Make the program self‑contained: include `#include <iostream>` and use `nvcc`‑compatible code.

All of this will be compiled as a CUDA C++ file (`.cu`). The code below follows these steps and can be compiled with `nvcc -o struct_example struct_example.cu` and run. 
*/

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

// Simple struct with two integer fields
struct Data {
    int val;
    int idx;
};

// CUDA error checking macro
#define CUDA_CHECK(err)                                 \
    do {                                                \
        cudaError_t err__ = (err);                      \
        if (err__ != cudaSuccess) {                     \
            std::cerr << "CUDA error: "                 \
                      << cudaGetErrorString(err__)      \
                      << " at line " << __LINE__       \
                      << " in file " << __FILE__ << std::endl; \
            exit(EXIT_FAILURE);                         \
        }                                               \
    } while (0)

// Kernel that modifies each struct in the array
__global__ void modify_struct(Data* d_data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        d_data[i].val = i * 10;  // Example modification
        d_data[i].idx = i;
    }
}

int main(int argc, char* argv[]) {
    // Determine N from command line or use default
    int N = 1024;
    if (argc > 1) {
        N = std::atoi(argv[1]);
        if (N <= 0) {
            std::cerr << "Invalid N. Using default 1024." << std::endl;
            N = 1024;
        }
    }

    std::cout << "Allocating " << N << " structs on device." << std::endl;

    // Allocate device memory
    Data* d_data = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_data, N * sizeof(Data)));

    // Launch kernel: one block of N threads if N <= 1024, else multiple blocks
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Launching kernel with " << blocks << " blocks of "
              << threadsPerBlock << " threads each." << std::endl;

    modify_struct<<<blocks, threadsPerBlock>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());          // Check launch error
    CUDA_CHECK(cudaDeviceSynchronize());     // Wait for kernel to finish

    // Copy result back to host
    Data* h_data = new Data[N];
    CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(Data),
                          cudaMemcpyDeviceToHost));

    // Verify a few elements
    std::cout << "Verifying some elements:" << std::endl;
    for (int i = 0; i < std::min(N, 10); ++i) {
        std::cout << "h_data[" << i << "] = { val = " << h_data[i].val
                  << ", idx = " << h_data[i].idx << " }" << std::endl;
    }

    // Clean up
    delete[] h_data;
    CUDA_CHECK(cudaFree(d_data));

    std::cout << "Done." << std::endl;
    return 0;
}
