/*
Try to use a C++ feature not supported by the targeted CUDA version (e.g., advanced templates or C++17 features without proper flags).
In this code I deliberately include a C++17 feature – std::optional – inside a CUDA kernel. This is not supported by older NVCC versions unless the compiler is invoked with the correct C++17 flags (e.g., -std=c++17). The goal is to demonstrate how using such a feature can cause a compilation failure when targeting a CUDA version that does not support the language standard. The rest of the code is minimal: a kernel that creates an optional<int> and prints its value, and a host main that launches the kernel. No additional libraries or complex templates are used beyond std::optional, which is a clear example of a modern C++ feature that may be unsupported in older CUDA toolchains.
*/

#include <cstdio>
#include <optional>

__global__ void kernel()
{
    // Using std::optional, a C++17 feature, inside a CUDA kernel.
    std::optional<int> opt = 42;
    if (opt)
    {
        printf("Optional value: %d\n", *opt);
    }
}

int main()
{
    // Launch the kernel
    kernel<<<1, 1>>>();
    // Synchronize and check for errors
    cudaDeviceSynchronize();
    return 0;
}
