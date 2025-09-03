```cpp
/*
Query and print the maximum number of threads per multiprocessor for the default device.
 
This program demonstrates how to query device properties using the CUDA Runtime API.
The main goal is to retrieve the maximum number of threads that can be resident on a single
multiprocessor (SM) for the currently selected device (the default device set by the system).
 
The approach is straightforward:
1. Obtain the current device id with cudaGetDevice().
2. Allocate a cudaDeviceProp structure and fill it using cudaGetDeviceProperties().
3. The desired value is stored in the field `maxThreadsPerMultiProcessor` of the structure.
4. Print the value to standard output.
 
We also include basic error checking: every CUDA API call is checked and if it fails, the
error message is printed and the program exits with a non-zero status. This ensures that
the program fails gracefully if, for example, no CUDA-capable device is present.
 
The code uses standard C++ I/O (iostream) for simplicity and portability across host
systems. It is fully self‑contained and can be compiled with `nvcc` to produce a CUDA
executable.
*/
#include <iostream>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error in file '" << __FILE__ << "' in line "   \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "."   \
                      << std::endl;                                            \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main() {
    int device = 0;
    // Get the current device (default device)
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    // Retrieve device properties
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // Print the maximum number of threads per multiprocessor
    std::cout << "Maximum threads per multiprocessor on device " << device
              << ": " << prop.maxThreadsPerMultiProcessor << std::endl;

    return 0;
}
```