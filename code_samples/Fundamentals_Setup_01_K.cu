```cpp
/*
Write a program that attempts to set the device to an invalid index (e.g., 999) and use proper error checking to report that the device does not exist.

Thinking process:
1. The goal is to demonstrate error handling when an invalid device index is requested.
2. We will call `cudaSetDevice(999)`, which is expected to fail because most systems will not have that many GPUs.
3. The CUDA runtime API returns an error code of type `cudaError_t`. We will check this return value.
4. If the return code is not `cudaSuccess`, we will use `cudaGetErrorString` to obtain a human-readable description.
5. The program will print the error message to standard output and exit with a non-zero status.
6. To keep the code minimal and clear, we will not launch any kernels or perform further GPU operations.
7. The code will be compiled with `nvcc` and thus must include the `<cuda_runtime.h>` header.
8. We will also include `<iostream>` for console output.
9. The program will be self‑contained in a single .cu file as requested.
10. All comments are placed inside the initial multiline comment block; the actual code follows afterward.
*/

#include <cuda_runtime.h>
#include <iostream>

int main() {
    const int invalidDeviceIndex = 999;
    cudaError_t err = cudaSetDevice(invalidDeviceIndex);

    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device to index " << invalidDeviceIndex
                  << ". Error: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    // If we reach here, the device index was valid (unlikely for 999).
    std::cout << "Successfully set CUDA device to index " << invalidDeviceIndex << std::endl;
    return EXIT_SUCCESS;
}
```