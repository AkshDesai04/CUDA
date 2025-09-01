```cpp
/*
Use `cudaDeviceSynchronize()` after the kernel launch. This is a common way to catch asynchronous errors from the kernel execution itself, not just launch configuration errors.

Thinking:
The goal is to demonstrate how `cudaDeviceSynchronize()` can be used to detect errors that occur during kernel execution, such as out‑of‑bounds memory accesses, which are not caught by `cudaGetLastError()` immediately after the kernel launch.  
The program will:
1. Allocate a small array on the device.
2. Launch a kernel that intentionally writes to an out‑of‑bounds index by launching more threads than the array size.
3. Check for launch‑time errors with `cudaGetLastError()` – this should return `cudaSuccess` because the kernel launch itself is valid.
4. Call `cudaDeviceSynchronize()` to force the host to wait for kernel completion. If the kernel executed incorrectly (e.g., performed an illegal memory write), `cudaDeviceSynchronize()` will return an error code.
5. Retrieve the error again with `cudaGetLastError()` after synchronization to confirm that an error was detected.
6. Report the error to the user and clean up.

The code includes a simple error‑checking macro for convenience and prints detailed error messages. It can be compiled with `nvcc` and run on any CUDA‑enabled device.
*/
#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in " << #call << ": "                      \
                      << cudaGetErrorString(err) << std::endl;                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Kernel that deliberately writes beyond the bounds of the array.
__global__ void faultyKernel(int *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // No bounds check: if idx >= n, this is an illegal memory access.
    arr[idx] = idx;
}

int main() {
    const int N = 10;                 // Size of the array on the device.
    const int threadsPerBlock = 5;    // Number of threads per block.
    const int blocks = 4;             // Number of blocks (total threads = 20 > N).

    int *d_arr = nullptr;
    CHECK_CUDA(cudaMalloc(&d_arr, N * sizeof(int)));

    // Launch the kernel with more threads than the array size.
    faultyKernel<<<blocks, threadsPerBlock>>>(d_arr, N);

    // Check for launch errors (synchronous errors).
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Launch error detected: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "No launch error detected." << std::endl;
    }

    // Synchronize to catch any asynchronous errors from kernel execution.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Runtime error detected after synchronization: "
                  << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "No runtime error detected after synchronization." << std::endl;
    }

    // Additional check for any errors that might have been set after sync.
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Post-synchronization error: " << cudaGetErrorString(err) << std::endl;
    }

    // Clean up.
    CHECK_CUDA(cudaFree(d_arr));
    return 0;
}
```