```cpp
/*
Write a program to get the ID of the currently active CUDA device using `cudaGetDevice` and print it.

Thinking:
1. We need to include the CUDA runtime header `cuda_runtime.h` to access `cudaGetDevice`.
2. Use `cudaGetDevice` which returns a `cudaError_t` and outputs the device ID via a pointer argument.
3. Handle the case where the call fails by printing the CUDA error string obtained via `cudaGetErrorString`.
4. If the call succeeds, print the device ID to the console.
5. Use standard C++ IO (`std::cout`) for printing.
6. Return `EXIT_SUCCESS` on success and `EXIT_FAILURE` on error.
7. Keep the program minimal and self‑contained, compiling with `nvcc`.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

int main() {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        std::cerr << "Error getting active CUDA device: " 
                  << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Active CUDA device ID: " << device << std::endl;
    return EXIT_SUCCESS;
}
```