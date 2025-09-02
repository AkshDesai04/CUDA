```cpp
/*
Aim of the program: Write a program that creates a summary table, printing the ID, Name, and Global Memory for every CUDA device found.

Thought Process:
1. The goal is to query all CUDA-capable devices on the system and display key information about each.
2. We will use the CUDA Runtime API, specifically `cudaGetDeviceCount` to determine how many devices are available, and `cudaGetDeviceProperties` to retrieve each device's properties.
3. The required information is:
   - Device ID (an integer starting from 0)
   - Device Name (a string provided by the CUDA API)
   - Global Memory (the amount of global device memory in bytes)
4. For a clean console output, we'll format the table using iomanip functions: `std::setw`, `std::left`, etc. This will create a neatly aligned table with headers.
5. Error handling is essential; if any CUDA call fails, we print a meaningful message and exit gracefully.
6. The program will compile with NVCC as a .cu file and run on any system with CUDA installed.

Implementation steps:
- Include `<iostream>`, `<iomanip>`, and `<cuda_runtime.h>`.
- In `main`, call `cudaGetDeviceCount`. If the call fails, print an error and exit.
- Iterate over device IDs from 0 to count-1.
- For each device, use `cudaGetDeviceProperties` to obtain `name` and `totalGlobalMem`.
- Print the table header once, then each row for a device.
- Return 0 on success.

This fulfills the requirement of producing a concise, self‑contained CUDA program that lists device IDs, names, and global memory sizes.
*/

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return EXIT_SUCCESS;
    }

    // Print table header
    std::cout << std::left
              << std::setw(5)  << "ID"
              << std::setw(30) << "Name"
              << std::setw(15) << "Global Memory (Bytes)"
              << std::endl;

    std::cout << std::string(5 + 30 + 15, '-') << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            std::cerr << "Error: cudaGetDeviceProperties failed for device " << dev
                      << ": " << cudaGetErrorString(err) << std::endl;
            continue; // skip this device
        }

        std::cout << std::left
                  << std::setw(5)  << dev
                  << std::setw(30) << prop.name
                  << std::setw(15) << prop.totalGlobalMem
                  << std::endl;
    }

    return EXIT_SUCCESS;
}
```