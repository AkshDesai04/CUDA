/*
Implement a kernel that adds a scalar value to every element of a vector: `output[i] = input[i] + scalar_value`.

**Thinking process:**

- The goal is to demonstrate a simple CUDA kernel that performs element‑wise addition of a scalar to each element of an input vector.
- Use a single kernel that accepts pointers to input and output arrays, the scalar value, and the array size.
- The kernel will compute a global index for each thread and, if it falls within bounds, write the result to the output array.
- On the host side:
  - Allocate host vectors for input and output.
  - Allocate device memory for input and output.
  - Copy input data from host to device.
  - Determine grid and block dimensions (e.g., 256 threads per block).
  - Launch the kernel.
  - Copy the output back to host.
  - Verify the results by printing both arrays.
- Include basic CUDA error checking using a macro to wrap CUDA API calls.
- Use `float` for the data type to keep it simple but could be any numeric type.
- The program is self‑contained, compiles with `nvcc`, and prints the results to standard output.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__    \
                      << ", code: " << err << " ("                       \
                      << cudaGetErrorString(err) << ")" << std::endl;       \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that adds a scalar to each element of the input vector
__global__ void add_scalar(const float *input, float *output, float scalar, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = input[idx] + scalar;
    }
}

int main()
{
    const int n = 10;
    const float scalar = 5.0f;

    // Host vectors
    std::vector<float> h_input(n);
    std::vector<float> h_output(n);

    // Initialize input vector
    for (int i = 0; i < n; ++i)
    {
        h_input[i] = static_cast<float>(i); // 0.0, 1.0, ..., 9.0
    }

    // Device pointers
    float *d_input = nullptr;
    float *d_output = nullptr;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_input, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_output, n * sizeof(float)));

    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // Determine kernel launch configuration
    const int threadsPerBlock = 256;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    add_scalar<<<blocks, threadsPerBlock>>>(d_input, d_output, scalar, n);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    // Print results
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Input vector:  ";
    for (float val : h_input)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "Scalar value:  " << scalar << std::endl;

    std::cout << "Output vector: ";
    for (float val : h_output)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
