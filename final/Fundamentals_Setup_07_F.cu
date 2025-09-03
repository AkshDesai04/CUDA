```cpp
/*
Aim of the program:
Query the device property `kernelExecTimeoutEnabled` and print whether a kernel execution timeout is enabled.

Thinking process:
1. Use CUDA runtime API to retrieve device properties via `cudaGetDeviceProperties`.
2. `cudaDeviceProp` struct contains the field `kernelExecTimeoutEnabled`, which is an integer flag (1 if enabled, 0 otherwise).
3. Determine the number of CUDA-capable devices with `cudaGetDeviceCount`.
4. For each device, call `cudaGetDeviceProperties`, check `kernelExecTimeoutEnabled`, and print a message indicating whether the timeout is enabled for that device.
5. Implement minimal error handling: check return codes of CUDA API calls and exit with an error message if a call fails.
6. Use `std::cout` for outputting results.
7. The program is written in C++ (CUDA C) and compiled with `nvcc`. No external libraries are required.
*/

#include <iostream>
#include <cuda_runtime.h>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        std::cerr << "Error: cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    if (deviceCount == 0)
    {
        std::cerr << "No CUDA-capable devices found." << std::endl;
        return EXIT_FAILURE;
    }

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess)
        {
            std::cerr << "Error: cudaGetDeviceProperties failed for device " << dev
                      << ": " << cudaGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "Device " << dev << " (" << prop.name << "): ";
        if (prop.kernelExecTimeoutEnabled)
            std::cout << "Kernel execution timeout is ENABLED." << std::endl;
        else
            std::cout << "Kernel execution timeout is DISABLED." << std::endl;
    }

    return EXIT_SUCCESS;
}
```