/*
Aim of the program: Try to use a C++ feature not supported by the targeted CUDA version (e.g., advanced templates or C++17 features without proper flags).

Thinking:
- The goal is to intentionally trigger a compilation failure on CUDA toolkits that do not support certain C++17 features.
- I'll use the `if constexpr` construct, which is a C++17 feature, inside a CUDA kernel template.
- The kernel will perform different operations based on whether the template type `T` is an integral type or not.
- The host code will launch the kernel for both `int` and `double` types to demonstrate both branches.
- If the CUDA compiler in the environment lacks C++17 support (e.g., older than CUDA 11), the code will fail to compile, as intended.
*/

#include <iostream>
#include <type_traits>
#include <cuda_runtime.h>

// Template kernel using C++17 if constexpr
template <typename T>
__global__ void testKernel(T* d, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // C++17 feature: constexpr if
        if constexpr (std::is_integral_v<T>)
            d[idx] = static_cast<T>(42);
        else
            d[idx] = static_cast<T>(3.1415);
    }
}

int main()
{
    const int N = 100;
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host memory for int and double
    int h_int[N];
    double h_double[N];

    // Allocate device memory
    int* d_int = nullptr;
    double* d_double = nullptr;
    cudaMalloc(&d_int, N * sizeof(int));
    cudaMalloc(&d_double, N * sizeof(double));

    // Launch kernel for int
    testKernel<int><<<blocks, threadsPerBlock>>>(d_int, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_int, d_int, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Launch kernel for double
    testKernel<double><<<blocks, threadsPerBlock>>>(d_double, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_double, d_double, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "First 5 results for int kernel: ";
    for (int i = 0; i < 5; ++i)
        std::cout << h_int[i] << " ";
    std::cout << std::endl;

    std::cout << "First 5 results for double kernel: ";
    for (int i = 0; i < 5; ++i)
        std::cout << h_double[i] << " ";
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_int);
    cudaFree(d_double);

    return 0;
}
