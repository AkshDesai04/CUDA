```cpp
/*
Write a kernel that modifies the data on the device (e.g., doubles every element) before you copy it back. Verify the result is as expected.

My thoughts while preparing this code:
1. I need a simple CUDA program that demonstrates how to copy data to the GPU, launch a kernel that modifies that data, and copy the result back.
2. The kernel should double each element of an array. I'll use a 1D grid of threads, each thread handling one array element.
3. I should allocate memory on the host, initialize it, allocate device memory, copy host data to device, launch the kernel, copy back, and then verify.
4. Verification can be done by checking that each element has indeed been doubled. For simplicity, I'll print the first few elements before and after to visually confirm.
5. I will include a small helper macro for error checking CUDA API calls to make debugging easier.
6. The program will be self-contained and should compile with nvcc without any external dependencies.
7. I'll keep the array size modest (e.g., 10 elements) for clear output, but the code will work for any size.
8. Comments in the code will explain each step, and the top multi-line comment includes the exact prompt text and my reasoning.
*/

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            std::cerr << "CUDA error in " << __FILE__              \
                      << " at line " << __LINE__ << ": "          \
                      << cudaGetErrorString(err) << std::endl;    \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Kernel that doubles each element of the array
__global__ void doubleElements(float *data, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        data[idx] *= 2.0f;
    }
}

int main()
{
    const size_t N = 10;                 // Number of elements
    const size_t bytes = N * sizeof(float);

    // Host array
    std::vector<float> h_data(N);
    for (size_t i = 0; i < N; ++i)
    {
        h_data[i] = static_cast<float>(i + 1); // 1.0, 2.0, ..., N
    }

    // Print original data
    std::cout << "Original data:\n";
    for (size_t i = 0; i < N; ++i)
        std::cout << h_data[i] << " ";
    std::cout << "\n";

    // Device memory
    float *d_data = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_data, bytes));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));

    // Launch kernel: choose block size of 256, grid size computed accordingly
    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    doubleElements<<<blocks, threadsPerBlock>>>(d_data, N);
    CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA(cudaDeviceSynchronize()); // Wait for kernel to finish

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost));

    // Print modified data
    std::cout << "Modified data (after doubling on device):\n";
    for (size_t i = 0; i < N; ++i)
        std::cout << h_data[i] << " ";
    std::cout << "\n";

    // Verify correctness
    bool correct = true;
    for (size_t i = 0; i < N; ++i)
    {
        float expected = (static_cast<float>(i + 1)) * 2.0f;
        if (h_data[i] != expected)
        {
            std::cerr << "Verification failed at index " << i
                      << ": expected " << expected
                      << ", got " << h_data[i] << std::endl;
            correct = false;
            break;
        }
    }
    if (correct)
        std::cout << "Verification succeeded: all values doubled correctly.\n";
    else
        std::cout << "Verification failed.\n";

    // Clean up
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}
```