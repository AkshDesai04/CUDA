/*
Aim of the program (word for word as requested):
Create a `__device__` function `add_two(int a, int b)` that returns `a+b`. Call it from your kernel.

Thinking:
- The goal is to demonstrate the use of a device function that is called from a kernel.  
- `add_two` is declared with `__device__` so it can only be executed on the GPU and can be invoked by kernels.  
- The kernel will accept input parameters `a` and `b`, call `add_two`, and write the result into global memory so the host can read it back.  
- For simplicity we will use a single-threaded kernel that writes one result, but the kernel is written to allow multiple threads if desired (by using the thread index).  
- Host code will allocate memory on the device, launch the kernel, copy the result back, and print it.  
- Basic error checking is added after each CUDA call for robustness.  
- The program is self‑contained and can be compiled with `nvcc` as a .cu file.  
*/

#include <cuda_runtime.h>
#include <iostream>

// Device function that simply adds two integers
__device__ int add_two(int a, int b) {
    return a + b;
}

// Kernel that calls the device function and stores the result in global memory
__global__ void add_kernel(int a, int b, int *result) {
    // For demonstration we use only one thread to compute the sum
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int sum = add_two(a, b);
        *result = sum;
    }
}

// Helper macro for checking CUDA errors
#define CHECK_CUDA(call)                                           \
    {                                                              \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            std::cerr << "CUDA error in " << __FILE__ << ":"       \
                      << __LINE__ << " : " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    }

int main() {
    const int a = 5;
    const int b = 7;
    int h_result = 0;
    int *d_result = nullptr;

    // Allocate device memory for the result
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(int)));

    // Launch kernel with a single block and single thread
    add_kernel<<<1, 1>>>(a, b, d_result);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA(cudaFree(d_result));

    // Output the result
    std::cout << "Result of add_two(" << a << ", " << b << ") = " << h_result << std::endl;

    return 0;
}
