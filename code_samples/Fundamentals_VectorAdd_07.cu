/*
Aim: Implement a kernel that squares every element in a vector: output[i] = input[i] * input[i].

Thinking:
To fulfill the requirement, the program will be written in CUDA C/C++ and compiled as a .cu file.  
The main components are:

1. **Kernel** – `squareKernel` receives a constant input array and a writable output array.  
   Each thread computes the square of a single element indexed by its global thread ID.  
   Bounds checking (`if (idx < N)`) guarantees safety for the last block when `N` is not a multiple of the block size.

2. **Host side** –  
   - Allocate host arrays and fill them with sample data.  
   - Allocate device memory with `cudaMalloc`.  
   - Copy the input array to device memory using `cudaMemcpy`.  
   - Launch the kernel with a suitable grid and block configuration (block size 256).  
   - Copy the output array back to host memory.  
   - Print the results to verify correctness.  
   - Clean up device and host resources.

3. **Error handling** – A helper macro `CUDA_CHECK` wraps CUDA API calls and aborts the program with a descriptive message if any call fails.  
   This ensures that any unexpected CUDA runtime error is reported early.

4. **Compile & Run** – The program can be compiled with `nvcc`:
   ```
   nvcc -o vector_square vector_square.cu
   ./vector_square
   ```
   The output will show the squared values for the sample input vector.

The code below is self‑contained, includes all necessary headers, and follows good CUDA programming practices.
*/

#include <iostream>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                              \
    do {                                                             \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess) {                                    \
            std::cerr << "CUDA error in " << __FILE__ << ":"         \
                      << __LINE__ << ": " << cudaGetErrorString(err)\
                      << std::endl;                                 \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

// Kernel that squares each element of the input vector
__global__ void squareKernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = input[idx];
        output[idx] = val * val;
    }
}

int main()
{
    const int N = 10;          // Size of the vector
    const int bytes = N * sizeof(float);

    // Allocate host memory
    float h_input[N];
    float h_output[N];

    // Initialize input vector with sample values
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i + 1); // 1.0, 2.0, ...
    }

    // Allocate device memory
    float *d_input = nullptr;
    float *d_output = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_input, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, bytes));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Determine grid and block dimensions
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Launch the kernel
    squareKernel<<<gridSize, blockSize>>>(d_input, d_output, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy the results back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    // Print the results
    std::cout << "Input  : ";
    for (int i = 0; i < N; ++i) std::cout << h_input[i] << " ";
    std::cout << "\nOutput : ";
    for (int i = 0; i < N; ++i) std::cout << h_output[i] << " ";
    std::cout << std::endl;

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
