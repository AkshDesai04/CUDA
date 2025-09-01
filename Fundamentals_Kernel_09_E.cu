/*
Create a `__host__ __device__` function (a function that can be called from both host and device). Call it from both `main` and your kernel.

Thinking:
- Need a function annotated __host__ __device__ that can be called from both host and device.
- In host main, we can call it directly and print the result.
- On device side, we need to create a kernel that calls the same function.
- The function can be simple like add two integers and return sum.
- The kernel can compute a result for each thread and store in an array.
- Allocate memory on host and device, launch kernel, copy back, print results.
- Ensure error checking for CUDA calls.
- Use __global__ void kernel(...) to launch kernel.
- Provide proper includes and definitions.
*/

#include <iostream>
#include <cuda_runtime.h>

#define N 10

// __host__ __device__ function that can be called from both host and device
__host__ __device__ int add(int a, int b) {
    return a + b;
}

// Kernel that uses the __host__ __device__ function
__global__ void addKernel(int *d_out, int b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_out[idx] = add(idx, b);
    }
}

// Utility macro for CUDA error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << "@" << __LINE__     \
                      << ": " << cudaGetErrorString(err) << std::endl;       \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main() {
    // Call __host__ __device__ function from host
    int hostResult = add(5, 7);
    std::cout << "Result of add(5, 7) on host: " << hostResult << std::endl;

    // Allocate host array
    int h_out[N];
    // Allocate device array
    int *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_out, N * sizeof(int)));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocks, threadsPerBlock>>>(d_out, 10, N);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure kernel completion

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print kernel results
    std::cout << "Kernel results (add(idx, 10)):" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "Index " << i << ": " << h_out[i] << std::endl;
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
