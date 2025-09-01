/*
Aim: Instead of `malloc`, use `new int[256]` for the host allocation.
Thoughts:
- The original task was to replace any usage of malloc with the C++ operator new for host memory allocation.
- In a CUDA program, host arrays can be allocated using `new int[size]` which returns a pointer that should be deallocated with `delete[]`.
- The rest of the CUDA logic remains unchanged: we allocate device memory with `cudaMalloc`, copy data with `cudaMemcpy`, launch a simple kernel, copy the result back, and then free all resources.
- For demonstration, a simple element‑wise addition kernel is used.
- Proper error checking is included for CUDA calls.
- At the end, host memory is freed with `delete[]` to match the new allocation.
*/

#include <iostream>
#include <cuda_runtime.h>

// Simple element‑wise addition kernel
__global__ void addKernel(const int* a, const int* b, int* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 256;
    const size_t size = N * sizeof(int);

    // Allocate host memory using new
    int* h_a = new int[N];
    int* h_b = new int[N];
    int* h_c = new int[N];

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = N - i;
    }

    // Device pointers
    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_a, size);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc d_a failed: " << cudaGetErrorString(err) << "\n"; return -1; }
    err = cudaMalloc((void**)&d_b, size);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc d_b failed: " << cudaGetErrorString(err) << "\n"; return -1; }
    err = cudaMalloc((void**)&d_c, size);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc d_c failed: " << cudaGetErrorString(err) << "\n"; return -1; }

    // Copy host data to device
    err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy to d_a failed: " << cudaGetErrorString(err) << "\n"; return -1; }
    err = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy to d_b failed: " << cudaGetErrorString(err) << "\n"; return -1; }

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // Copy result back to host
    err = cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy to h_c failed: " << cudaGetErrorString(err) << "\n"; return -1; }

    // Verify result (simple print)
    std::cout << "Result (first 10 elements):\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << h_c[i] << " ";
    }
    std::cout << "\n";

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory allocated with new
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
