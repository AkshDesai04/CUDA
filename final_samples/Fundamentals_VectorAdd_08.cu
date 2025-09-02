```cpp
/*
Aim of the program: 
Implement a kernel that adds a scalar value to every element of a vector: `output[i] = input[i] + scalar_value`.

Thinking process:
1. The program must demonstrate the use of CUDA to perform a simple element‑wise addition of a scalar to each element of an array.
2. Host side:
   - Create a host vector of a given size, fill it with test values.
   - Allocate device memory for input and output arrays.
   - Copy input data from host to device.
   - Define a scalar value to be added.
3. Device side:
   - Write a CUDA kernel that takes input array, output array, scalar, and size as arguments.
   - Each thread calculates its global index and, if within bounds, performs the addition.
4. Kernel launch:
   - Choose a reasonable block size (e.g., 256 threads per block) and compute the number of blocks.
   - Launch the kernel with those dimensions.
5. After kernel execution:
   - Copy the result back to host memory.
   - Optionally, verify correctness by printing a few values.
6. Clean up device memory and exit.
7. Add a simple error‑checking macro for CUDA calls to make debugging easier.

The final code below follows this plan and is ready to compile with `nvcc` to produce a `.cu` file.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Simple CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " code=" << err << " \"" << cudaGetErrorString(err) \
                      << "\"\n";                                              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel that adds a scalar to each element of the input array
__global__ void add_scalar_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  float scalar,
                                  size_t n) {
    // Calculate global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] + scalar;
    }
}

int main() {
    const size_t N = 1 << 20;          // 1M elements
    const float scalar_value = 5.0f;   // Example scalar

    // Allocate and initialize host vectors
    std::vector<float> h_input(N, 1.0f);  // Fill with 1.0 for easy verification
    std::vector<float> h_output(N, 0.0f);

    // Device pointers
    float *d_input = nullptr;
    float *d_output = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, N * sizeof(float)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threads_per_block = 256;
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    add_scalar_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, scalar_value, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Optional: verify a few results
    bool ok = true;
    for (int i = 0; i < 10; ++i) {
        float expected = h_input[i] + scalar_value;
        if (fabs(h_output[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": got " << h_output[i]
                      << ", expected " << expected << "\n";
            ok = false;
            break;
        }
    }
    if (ok) {
        std::cout << "Kernel executed successfully. First 10 results: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << h_output[i] << " ";
        }
        std::cout << "\n";
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
```