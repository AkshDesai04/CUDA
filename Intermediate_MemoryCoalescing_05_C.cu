/*
Aim of the program: Implement the SoA version: create float* d_xs, *d_ys, *d_zs. Write a kernel that reads from d_xs: output[i] = d_xs[i].

My thinking:
- The problem asks for a CUDA program that uses a Structure of Arrays (SoA) layout for three float arrays: xs, ys, zs. 
- We need device pointers d_xs, d_ys, d_zs that will be allocated on the GPU. 
- The kernel must simply copy each element of d_xs to an output array. 
- For a complete example, we also allocate a host array for xs, ys, zs, initialize them with sample data, copy them to the device, launch the kernel, copy the output back, and display the results. 
- Error checking will be added through a helper macro to simplify CUDA API error handling. 
- The program is self‑contained in a single .cu file, uses standard CUDA runtime API functions (cudaMalloc, cudaMemcpy, cudaFree, etc.), and prints a few elements of the output to verify correctness. 
- The kernel uses a simple one‑dimensional grid/block configuration and checks bounds before writing to avoid out‑of‑range accesses. 
- This code can be compiled with nvcc and run on any CUDA capable GPU. 
*/

#include <iostream>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err_) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel that copies each element from d_xs to output
__global__ void copyX(const float *d_xs, float *d_output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_output[idx] = d_xs[idx];
    }
}

int main() {
    const int N = 10;                     // Number of elements
    const int sizeBytes = N * sizeof(float);

    // Allocate and initialize host arrays
    float h_xs[N], h_ys[N], h_zs[N];
    for (int i = 0; i < N; ++i) {
        h_xs[i] = static_cast<float>(i) + 0.1f;          // e.g., 0.1, 1.1, ...
        h_ys[i] = static_cast<float>(i) + 0.2f;          // e.g., 0.2, 1.2, ...
        h_zs[i] = static_cast<float>(i) + 0.3f;          // e.g., 0.3, 1.3, ...
    }

    // Device pointers
    float *d_xs = nullptr, *d_ys = nullptr, *d_zs = nullptr, *d_output = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_xs, sizeBytes));
    CUDA_CHECK(cudaMalloc(&d_ys, sizeBytes));
    CUDA_CHECK(cudaMalloc(&d_zs, sizeBytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeBytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_xs, h_xs, sizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, h_ys, sizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_zs, h_zs, sizeBytes, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel to copy xs to output
    copyX<<<blocksPerGrid, threadsPerBlock>>>(d_xs, d_output, N);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to finish

    // Copy result back to host
    float h_output[N];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeBytes, cudaMemcpyDeviceToHost));

    // Print the output to verify
    std::cout << "Output from device (copied from d_xs):" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "h_output[" << i << "] = " << h_output[i] << std::endl;
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_zs));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
