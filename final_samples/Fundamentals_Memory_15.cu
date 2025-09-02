/*
Declare a global device variable using `__device__`. Write a kernel that modifies this variable. Write another kernel that reads it. Demonstrate how its value persists across kernel launches within the same application.

Thinking:

1. We need a global variable that exists on the device side and is shared by all kernels launched during the life of the application.  
2. The simplest type to illustrate persistence is an `int`.  
3. `__device__ int d_global = 0;` declares it.  
4. The first kernel will write a value to this variable.  
5. The second kernel will read the value and write it to a device array that the host can read back.  
6. The host program will launch the first kernel, then the second kernel, copy the result back and print it.  
7. To demonstrate persistence, we launch the first kernel twice with different values and read back after each launch.  
8. The code is written in standard CUDA C/C++ and is self‑contained; compile with `nvcc -arch=sm_52 -o demo demo.cu`.  
9. Error checking is included via a helper macro for clarity.  
10. The program is a single `.cu` file with no external dependencies beyond the CUDA runtime and the standard library.
*/

#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                        \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " : "\
                      << cudaGetErrorString(err) << std::endl;                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

// Global device variable
__device__ int d_global = 0;

// Kernel to set the global variable
__global__ void set_global_value(int val) {
    // Only one thread needed
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_global = val;
    }
}

// Kernel to read the global variable
__global__ void get_global_value(int *out) {
    // Only one thread needed
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = d_global;
    }
}

int main() {
    const int size = sizeof(int);
    int h_val;
    int *d_val;

    // Allocate device memory for reading the global variable
    CHECK_CUDA(cudaMalloc((void **)&d_val, size));

    // First launch: set to 42
    set_global_value<<<1, 1>>>(42);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Read back
    get_global_value<<<1, 1>>>(d_val);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_val, d_val, size, cudaMemcpyDeviceToHost));
    std::cout << "After first set (42), global value = " << h_val << std::endl;

    // Second launch: set to 7
    set_global_value<<<1, 1>>>(7);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Read back
    get_global_value<<<1, 1>>>(d_val);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_val, d_val, size, cudaMemcpyDeviceToHost));
    std::cout << "After second set (7), global value = " << h_val << std::endl;

    // Clean up
    CHECK_CUDA(cudaFree(d_val));

    return 0;
}
