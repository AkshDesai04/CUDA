```cpp
/* 
Aim of the program: Create a helper C++ function `void launch_kernel_1d(int n_elements)` that computes the correct grid/block dimensions and launches a kernel.

Thinking:
- The task is to provide a helper function that can be reused to launch a simple 1D kernel with optimal grid and block configuration.
- I choose a block size of 256 threads, a common choice that works well on most GPUs. This can be easily adjusted if needed.
- The grid size is calculated as `(n_elements + blockDim - 1) / blockDim` to cover all elements.
- For demonstration, the kernel will write the global thread index into an array, effectively storing indices 0..n_elements-1.
- The helper will allocate device memory, launch the kernel, copy the result back to host, print a few values for verification, and then clean up.
- The code is self-contained and can be compiled with nvcc. It includes error checking after kernel launch to ensure proper execution.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void dummy_kernel(int *data, int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        data[idx] = idx;  // store index for verification
    }
}

void launch_kernel_1d(int n_elements) {
    const int blockSize = 256;  // choose a sensible default block size
    int gridSize = (n_elements + blockSize - 1) / blockSize;

    // Allocate device memory
    int *d_data = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_data, n_elements * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Launch kernel
    dummy_kernel<<<gridSize, blockSize>>>(d_data, n_elements);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        return;
    }

    // Copy result back to host
    std::vector<int> h_data(n_elements);
    err = cudaMemcpy(h_data.data(), d_data, n_elements * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        return;
    }

    // Verify by printing first few elements
    std::cout << "First 10 elements after kernel execution:" << std::endl;
    for (int i = 0; i < std::min(n_elements, 10); ++i) {
        std::cout << "h_data[" << i << "] = " << h_data[i] << std::endl;
    }

    // Clean up
    cudaFree(d_data);
}

int main() {
    int n = 1000;  // example number of elements
    launch_kernel_1d(n);
    return 0;
}
```