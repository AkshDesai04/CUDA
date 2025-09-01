/*
Aim: Query the device property `kernelExecTimeoutEnabled` and print whether a kernel execution timeout is enabled.

Thinking:
The goal is to write a minimal CUDA C++ program that interrogates the properties of a CUDA-capable device to determine whether the kernel execution timeout feature is enabled. 
The relevant property resides in the `cudaDeviceProp` structure returned by `cudaGetDeviceProperties`. Specifically, the `kernelExecTimeoutEnabled` field is a boolean flag that indicates whether the operating system (typically Windows) enforces a timeout on GPU kernels. If this flag is true, kernels that run longer than a certain threshold will be terminated by the OS to avoid freezing the system.

To implement this:
1. Include the necessary headers: `<cuda_runtime.h>` for CUDA runtime API, `<iostream>` for console output.
2. In `main`, obtain the number of devices with `cudaGetDeviceCount`. If none are found, print an error and exit.
3. For simplicity, query the first device (index 0) using `cudaGetDeviceProperties`.
4. Inspect the `kernelExecTimeoutEnabled` field. If true, print "Kernel execution timeout is enabled." Otherwise, print "Kernel execution timeout is disabled."
5. Include error checking for CUDA calls: if any call fails, report the CUDA error string and exit.

The code below follows this plan and can be compiled with `nvcc` to produce a runnable executable.*/

#include <cuda_runtime.h>
#include <iostream>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to get device count: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    if (deviceCount == 0)
    {
        std::cerr << "No CUDA-capable devices found." << std::endl;
        return EXIT_FAILURE;
    }

    // Query the first device (index 0)
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Device 0: " << prop.name << std::endl;
    std::cout << "Kernel execution timeout is "
              << (prop.kernelExecTimeoutEnabled ? "ENABLED" : "DISABLED")
              << "." << std::endl;

    return EXIT_SUCCESS;
}
