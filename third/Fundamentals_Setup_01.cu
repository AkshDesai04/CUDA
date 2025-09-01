/*
Aim: Write a CUDA program to query and print the number of CUDA-enabled devices available on your system. The program should use the `cudaGetDeviceCount` API call and print a user-friendly message, such as 'Found X CUDA-capable devices.'

Thinking:
1. The program is a simple host-only CUDA application that queries the number of available GPU devices.
2. We need to include the CUDA runtime header <cuda_runtime.h> to access `cudaGetDeviceCount`.
3. We'll use a minimal `main` function that:
   - Declares an integer to hold the device count.
   - Calls `cudaGetDeviceCount(&count)` and checks the returned `cudaError_t`.
   - If the call fails, we print an error message using `cudaGetErrorString(err)`.
   - If successful, we print "Found X CUDA-capable devices." where X is the device count.
4. The program is straightforward; no device kernels are launched.
5. We'll use `std::cout` for printing, so we include `<iostream>`.
6. The code will compile with `nvcc`, e.g., `nvcc -o device_count device_count.cu`.
7. Edge cases:
   - If there are zero devices, the message will correctly say "Found 0 CUDA-capable devices."
   - If `cudaGetDeviceCount` returns an error, we report it and exit with non-zero status.
*/

#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Error querying device count: " 
                  << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Found " << deviceCount 
              << " CUDA-capable device" << (deviceCount == 1 ? "" : "s") << "." << std::endl;

    return 0;
}
