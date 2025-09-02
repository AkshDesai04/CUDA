```cpp
/*
Aim of the program: Implement matrix-scalar addition.

Thinking:
1. The problem is straightforward: add a scalar value to each element of a matrix.
2. In CUDA, we will treat the matrix as a 1D array in row-major order.
3. We'll create a kernel that each thread handles one element: d_matrix[idx] += scalar.
4. The host code will:
   - Define matrix dimensions (e.g., 4x5).
   - Allocate host memory and initialize it with sample values.
   - Allocate device memory and copy the matrix to the device.
   - Launch the kernel with an appropriate grid/block configuration.
   - Copy the result back to host memory.
   - Print the resulting matrix to verify correctness.
5. Error checking will be minimal for brevity but will capture CUDA API errors.
6. The program will compile as a .cu file and can be built with nvcc.
*/

#include <iostream>
#include <cuda_runtime.h>

// Kernel to add a scalar to each element of the matrix
__global__ void addScalarKernel(float* d_matrix, float scalar, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        d_matrix[idx] += scalar;
    }
}

// Helper macro for checking CUDA errors
#define CUDA_CHECK(err)                                                   \
    if (err != cudaSuccess) {                                             \
        std::cerr << "CUDA error (" << err << "): " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE);                                               \
    }

int main() {
    // Matrix dimensions
    const int rows = 4;
    const int cols = 5;
    const int numElements = rows * cols;

    // Host memory allocation
    float h_matrix[numElements];

    // Initialize matrix with some values
    for (int i = 0; i < numElements; ++i) {
        h_matrix[i] = static_cast<float>(i);  // Example: 0, 1, 2, ...
    }

    // Scalar to add
    float scalar = 2.5f;

    // Device memory allocation
    float* d_matrix = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_matrix, numElements * sizeof(float)));

    // Copy matrix from host to device
    CUDA_CHECK(cudaMemcpy(d_matrix, h_matrix, numElements * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel: one thread per element
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    addScalarKernel<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, scalar, numElements);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_matrix, d_matrix, numElements * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_matrix));

    // Print the resulting matrix
    std::cout << "Resulting matrix after adding scalar " << scalar << ":\n";
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::cout << h_matrix[r * cols + c] << "\t";
        }
        std::cout << "\n";
    }

    return 0;
}
```