/*
Implement a kernel that adds a scalar value to every element of a vector: `output[i] = input[i] + scalar_value`.

Thinking process:
1. The core of the program is a CUDA kernel that takes an input vector, an output vector, a scalar value, and the length of the vector. Each thread will handle one element: `output[i] = input[i] + scalar`.
2. In `main()` we allocate a small test vector on the host, initialize it with known values, and set a scalar.
3. Allocate corresponding device memory using `cudaMalloc`, copy the input vector to device memory with `cudaMemcpy`.
4. Determine a suitable launch configuration (block size 256, grid size computed to cover all elements). Launch the kernel.
5. Copy the result back to the host with `cudaMemcpy`.
6. Print a few elements to verify the addition.
7. Clean up device memory.
8. Add basic error checking after CUDA calls to aid debugging.
9. The program is self‑contained and can be compiled with `nvcc`. It demonstrates memory transfer, kernel launch, and simple vector arithmetic.

*/

#include <iostream>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(err)                                                    \
    do {                                                                   \
        cudaError_t err_ = (err);                                          \
        if (err_ != cudaSuccess) {                                        \
            std::cerr << "CUDA error: " << cudaGetErrorString(err_)       \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";    \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

// Kernel: adds a scalar to each element of the input vector
__global__ void add_scalar(const float* input, float* output, float scalar, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        output[idx] = input[idx] + scalar;
}

int main()
{
    const int N = 1024;            // Vector size
    const float scalar = 5.0f;     // Value to add

    // Allocate host memory
    float* h_input  = new float[N];
    float* h_output = new float[N];

    // Initialize input vector with values 0, 1, 2, ...
    for (int i = 0; i < N; ++i)
        h_input[i] = static_cast<float>(i);

    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_input,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, N * sizeof(float)));

    // Copy input from host to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_scalar<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, scalar, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify results (print first 10 elements)
    std::cout << "First 10 results of adding scalar " << scalar << ":\n";
    for (int i = 0; i < 10; ++i)
        std::cout << h_input[i] << " + " << scalar << " = " << h_output[i] << '\n';

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    delete[] h_input;
    delete[] h_output;

    return 0;
}
