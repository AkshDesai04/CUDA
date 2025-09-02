/*
Aim of the program: Implement a kernel that computes `x^n` where `n` is passed as a parameter: `output[i] = powf(input[i], n)`.

Thinking:
To fulfill the requirement, I will write a complete CUDA C program that:
1. Declares a GPU kernel `powKernel` that takes an input array, an output array, a float `n`, and the array size `N`. Each thread computes the power of the element at its global index using `powf`.
2. In `main`, I will:
   - Define a small test dataset (e.g., N = 10) with known input values.
   - Allocate device memory for the input and output arrays.
   - Copy the input array from host to device.
   - Launch the kernel with a suitable grid and block configuration.
   - Copy the output back to the host.
   - Print the results for verification.
3. Include error checking for CUDA API calls to catch and report any failures.
4. Use standard headers (`<cuda_runtime.h>`, `<iostream>`, `<cmath>`).
5. Keep the program self‑contained and compile‑ready as a `.cu` file.
6. Ensure that the code follows best practices: proper cleanup, synchronization, and descriptive variable names.

The resulting file is ready to compile with `nvcc` and run, demonstrating the power calculation kernel.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in file '" << __FILE__                  \
                      << "' in line " << __LINE__ << ": "                   \
                      << cudaGetErrorString(err) << std::endl;              \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

// Kernel that computes output[i] = powf(input[i], n)
__global__ void powKernel(const float *input, float *output, float n, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = powf(input[idx], n);
    }
}

int main() {
    const int N = 10;               // Number of elements
    const float power = 3.0f;       // Example exponent n

    // Host input data
    std::vector<float> h_input(N);
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i) + 1.0f;  // 1.0, 2.0, ..., 10.0
    }

    std::vector<float> h_output(N, 0.0f);

    // Device pointers
    float *d_input = nullptr;
    float *d_output = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_output, N * sizeof(float)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    powKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, power, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    std::cout << "Input\t" << "Output (" << power << "th power)" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_input[i] << "\t" << h_output[i] << std::endl;
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
