```cpp
/*
Use `cudaMemcpyToSymbol` and `cudaMemcpyFromSymbol` to initialize and read the `__device__` variable from the host.

Thinking:
The goal is to demonstrate how host code can set a global device variable using cudaMemcpyToSymbol, launch a kernel that operates on it, and then retrieve its value back to the host using cudaMemcpyFromSymbol. 
We define a simple __device__ integer, set it to a known value on the host, launch a kernel that increments it, copy it back to the host, and print the result. 
Error checking is included after each CUDA call. 
*/

#include <iostream>
#include <cuda_runtime.h>

// Global device variable
__device__ int devVar;

// Kernel that modifies the device variable
__global__ void incKernel() {
    devVar += 10;
}

// Helper macro for error checking
#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA error at " << __FILE__         \
                      << ":" << __LINE__ << " - "             \
                      << cudaGetErrorString(err) << std::endl;\
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

int main() {
    int initVal = 100;
    int hostVar = 0;

    // Initialize the device variable from the host
    CHECK_CUDA(cudaMemcpyToSymbol(devVar, &initVal, sizeof(int),
                                  0, cudaMemcpyHostToDevice));

    // Launch the kernel
    incKernel<<<1,1>>>();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Read the updated value back to the host
    CHECK_CUDA(cudaMemcpyFromSymbol(&hostVar, devVar, sizeof(int),
                                    0, cudaMemcpyDeviceToHost));

    std::cout << "Value of devVar after kernel execution: " << hostVar << std::endl;

    return 0;
}
```