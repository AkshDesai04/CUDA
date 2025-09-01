```cpp
/* 
Aim of the program: 
Write a generic device_copy function that takes two void* pointers, a size, and calls cudaMemcpy with the DtoD kind. 
Entire thinking: 
The goal is to provide a simple, reusable CUDA helper that performs a device-to-device memory copy using cudaMemcpy. 
Because the function operates on void* pointers, it can accept pointers to any type of device memory, which makes it generic. 
We need to include the CUDA runtime header <cuda_runtime.h> and wrap the cudaMemcpy call so that any errors can be reported in a straightforward way. 
To make debugging easier, we’ll define a macro CUDA_CHECK that checks the return status of CUDA API calls and prints an error message if something goes wrong. 
The device_copy function itself will be marked __host__ (the default) and will not be compiled for device code, as it only invokes a runtime API. 
The function signature is: 
    void device_copy(void* dst, void* src, size_t size); 
Inside the function, we call cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice); 
We also call CUDA_CHECK on the result to ensure we catch errors. 
A small test in main will allocate two device arrays, copy data from one to the other using device_copy, and then copy back to host to verify correctness. 
This demonstrates that the function works and also provides a minimal example for compilation. 
The code is self-contained and can be compiled with nvcc. 
*/

#include <iostream>
#include <cuda_runtime.h>

// Helper macro to check CUDA API call results
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__     \
                      << " : " << cudaGetErrorString(err) << std::endl;      \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Generic device-to-device copy function
void device_copy(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}

// Simple kernel to initialize data on device
__global__ void init_kernel(int* arr, int value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = value;
}

int main() {
    const int N = 10;
    size_t bytes = N * sizeof(int);

    int *d_src = nullptr;
    int *d_dst = nullptr;
    int *h_result = new int[N];

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));

    // Initialize source array on device
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    init_kernel<<<blocks, threadsPerBlock>>>(d_src, 42, N);
    CUDA_CHECK(cudaGetLastError());

    // Perform device-to-device copy using our generic function
    device_copy(d_dst, d_src, bytes);

    // Copy result back to host for verification
    CUDA_CHECK(cudaMemcpy(h_result, d_dst, bytes, cudaMemcpyDeviceToHost));

    // Verify
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_result[i] != 42) {
            ok = false;
            break;
        }
    }

    std::cout << "Copy " << (ok ? "succeeded" : "failed") << std::endl;

    // Clean up
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    delete[] h_result;

    return 0;
}
```