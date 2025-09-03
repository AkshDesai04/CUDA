/*
Implement `output[i] = sinf(input[i])`.

My thinking:
To fulfill the task, I need to write a CUDA program that applies the sine function to each element of an input array and stores the result in an output array. The simplest way is to write a CUDA kernel that computes `sinf(input[i])` for each index `i`.  
The kernel will calculate a global thread index from block and thread indices, check that the index is within the bounds of the array, and then write the result to the output array.

On the host side, the program will:
1. Allocate a host array of a chosen size `N` and initialize it with some values (e.g., a linear progression from 0 to 2π).
2. Allocate device memory for both input and output arrays.
3. Copy the host input array to the device.
4. Launch the kernel with a suitable grid and block size (e.g., 256 threads per block).
5. Copy the result back to the host.
6. Print a few results to verify correctness.
7. Clean up all allocated memory.

I will also include a simple error‑checking macro to handle CUDA API errors.  
The program is self‑contained and should compile with `nvcc` into a .cu file. It demonstrates basic CUDA memory management, kernel launch, and the use of the `sinf` function from `<math.h>`.*/

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__    \
                      << " - " << cudaGetErrorString(err) << std::endl;     \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// CUDA kernel that computes sinf(input[i]) and writes to output[i]
__global__ void sinfKernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = sinf(input[idx]);
    }
}

int main() {
    const int N = 1024;                    // Number of elements
    const int SIZE = N * sizeof(float);    // Size in bytes

    // Allocate host memory
    float* h_input  = (float*)malloc(SIZE);
    float* h_output = (float*)malloc(SIZE);
    if (!h_input || !h_output) {
        std::cerr << "Failed to allocate host memory." << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize host input with values ranging from 0 to 2π
    for (int i = 0; i < N; ++i) {
        h_input[i] = 2.0f * static_cast<float>(M_PI) * i / N;
    }

    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_input,  SIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_output, SIZE));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, SIZE, cudaMemcpyHostToDevice));

    // Launch kernel: 256 threads per block
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    sinfKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, SIZE, cudaMemcpyDeviceToHost));

    // Print first 10 results for verification
    std::cout << "Index\tInput\t\tOutput(sinf)" << std::endl;
    for (int i = 0; i < 10 && i < N; ++i) {
        std::cout << i << "\t" << h_input[i] << "\t" << h_output[i] << std::endl;
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return EXIT_SUCCESS;
}
