```cpp
/*
Aim: Implement a kernel that cubes every element: output[i] = input[i] * input[i] * input[i].

Thinking:
1. The requirement is straightforward: for each element in an input array, compute its cube and store it in an output array.
2. To achieve parallelism, we assign one thread per element. The kernel will be simple: compute the global index, check bounds, and perform the cube.
3. For the launch configuration, we choose a reasonable block size (e.g., 256 threads per block) and calculate the number of blocks needed to cover the entire array.
4. Host code responsibilities:
   - Allocate and initialize a host array with sample data.
   - Allocate device memory for input and output arrays.
   - Copy the input array from host to device.
   - Launch the kernel with the calculated grid/block dimensions.
   - Copy the output array back to host.
   - Verify the results by printing both input and output arrays.
   - Clean up device memory.
5. Error handling is added via a simple macro `CUDA_CHECK` to catch and report any CUDA API errors.
6. The program is self‑contained and can be compiled with `nvcc example.cu -o example` and executed on a machine with CUDA installed.
*/

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Macro to check CUDA API calls
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " : " \
                      << cudaGetErrorString(err) << std::endl;                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Kernel to cube each element
__global__ void cubeKernel(const float *input, float *output, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = input[idx];
        output[idx] = val * val * val;
    }
}

int main()
{
    const size_t N = 10; // Example size; can be modified
    std::vector<float> h_input(N);
    std::vector<float> h_output(N, 0.0f);

    // Initialize input with sample values
    for (size_t i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // Device pointers
    float *d_input = nullptr;
    float *d_output = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, N * sizeof(float)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    cubeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy output data back to host
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    std::cout << "Index\tInput\tCube\n";
    for (size_t i = 0; i < N; ++i) {
        std::cout << i << "\t" << h_input[i] << "\t" << h_output[i] << "\n";
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
```