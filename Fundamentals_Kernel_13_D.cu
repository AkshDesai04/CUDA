```cuda
/*
Aim of the program:
Modify the kernel to take a fourth argument, `int scalar`, and compute `b[i] = a[i] + scalar;`.

Thinking:
To satisfy this requirement, I will modify the existing CUDA kernel to accept an additional integer parameter called `scalar`. The kernel will iterate over the array indices and add this scalar value to each element of the input array `a`, storing the result in the output array `b`. I will also update the host code to allocate memory for the input and output arrays, copy the data to the device, launch the kernel with the new argument, and copy the result back to the host. Basic error checking will be included for robustness. The program will use a simple test case to demonstrate the functionality.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " : " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Kernel that adds a scalar to each element of array a and writes to array b
__global__ void addScalarKernel(const int* a, int* b, int n, int scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        b[idx] = a[idx] + scalar;
    }
}

int main() {
    const int N = 1 << 20; // 1,048,576 elements
    const int scalar = 5;  // value to add

    // Allocate host memory
    std::vector<int> h_a(N, 1); // initialize with 1s
    std::vector<int> h_b(N, 0);

    // Allocate device memory
    int *d_a = nullptr;
    int *d_b = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_a, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, N * sizeof(int)));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // Kernel launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    addScalarKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N, scalar);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_b.data(), d_b, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify a few elements
    bool ok = true;
    for (int i = 0; i < 10; ++i) {
        if (h_b[i] != h_a[i] + scalar) {
            std::cerr << "Mismatch at index " << i << ": "
                      << h_b[i] << " != " << h_a[i] + scalar << std::endl;
            ok = false;
            break;
        }
    }
    if (ok) {
        std::cout << "Success: first 10 elements are correct." << std::endl;
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    return 0;
}
```