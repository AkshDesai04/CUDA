/*
Aim: Try to use a C++ feature not supported by the targeted CUDA version (e.g., advanced templates or C++17 features without proper flags).

My approach is to incorporate a C++17 feature, `if constexpr`, inside a device function, which older CUDA compilers might not support unless compiled with `-std=c++17` or higher. I also include a templated lambda (generic lambda syntax, C++14/C++20) that uses the `<typename T>` syntax, which older CUDA setups will reject because they do not support C++14 or later features by default. The code is intentionally written to fail on older CUDA setups because it uses these features. I further call a device function from the host to demonstrate how calling device code from host is not allowed without proper CUDA execution context, which will also cause a compile-time error. The goal is to illustrate the pitfalls of using unsupported C++ features in CUDA code.
*/

#include <iostream>
#include <cuda_runtime.h>

__global__ void kernel() {
    // Simple kernel does nothing
}

// C++17 if constexpr in a device function
template<bool B>
__device__ void constexprTest() {
    if constexpr (B) {
        printf("Condition is true\n");
    } else {
        printf("Condition is false\n");
    }
}

// C++14/C++20 generic lambda
auto genericLambda = []<typename T>(T x) -> T {
    return x * x;
};

int main() {
    // Launch kernel
    kernel<<<1,1>>>();
    cudaDeviceSynchronize();

    // Attempt to call constexprTest from host (will not compile on older CUDA)
    constexprTest<true>();
    constexprTest<false>();

    // Use the generic lambda on host
    int val = genericLambda(5);
    std::cout << "genericLambda(5) = " << val << std::endl;

    return 0;
}
