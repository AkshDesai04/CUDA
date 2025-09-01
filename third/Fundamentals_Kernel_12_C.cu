```cpp
/*
Aim of the program:
Create a helper C++ function `void launch_kernel_1d(int n_elements)` that computes the correct grid/block dimensions and launches a kernel.

Thinking and design:
1. A 1‑D kernel that operates over `n_elements` elements is required.  
2. For performance a typical block size of 256 threads is chosen.  
3. The grid dimension is computed as the ceiling of `n_elements / blockDim.x`.  
4. The helper function will allocate a device array, launch the kernel, copy the result back, and print the first few elements for verification.  
5. A simple error‑checking macro `CUDA_CHECK` is used for clarity.  
6. The kernel `compute_kernel` simply writes the global thread index into the array so that we can easily confirm the launch.  
7. `main` demonstrates the helper by launching it with a sample size.

The code below is a complete `.cu` file that can be compiled with `nvcc`.
*/

#include <iostream>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__    \
                      << " - " << cudaGetErrorString(err) << std::endl;     \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Simple kernel that writes the global thread index into the array
__global__ void compute_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx;
    }
}

/*
 * Helper function that calculates grid and block dimensions and launches the kernel.
 */
void launch_kernel_1d(int n_elements) {
    // Choose a reasonable block size
    const int blockSize = 256;

    // Compute grid size so that all elements are covered
    int gridSize = (n_elements + blockSize - 1) / blockSize;

    // Allocate device memory
    int *d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, n_elements * sizeof(int)));

    // Launch the kernel
    compute_kernel<<<gridSize, blockSize>>>(d_data, n_elements);

    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    int *h_data = new int[n_elements];
    CUDA_CHECK(cudaMemcpy(h_data, d_data, n_elements * sizeof(int), cudaMemcpyDeviceToHost));

    // Simple output: print first 10 elements
    std::cout << "First 10 elements of the array after kernel launch:" << std::endl;
    for (int i = 0; i < std::min(10, n_elements); ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    delete[] h_data;
    CUDA_CHECK(cudaFree(d_data));
}

int main() {
    int n = 1000;  // Example size
    launch_kernel_1d(n);
    return 0;
}
```