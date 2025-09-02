/*
 * Query the device property `kernelExecTimeoutEnabled` and print whether a kernel execution timeout is enabled.
 *
 * The aim of this program is to demonstrate how to use the CUDA Runtime API to retrieve device properties
 * and examine a specific boolean field: `kernelExecTimeoutEnabled`. This property indicates whether
 * the system enforces a timeout on kernel execution (typically relevant for laptops where the GPU
 * may be integrated with a display driver that requires the kernel to finish within a certain
 * time window). The program will:
 *
 * 1. Determine the number of CUDA-capable devices.
 * 2. For each device, retrieve its properties using `cudaGetDeviceProperties`.
 * 3. Print out the device name and the status of `kernelExecTimeoutEnabled` (Enabled/Disabled).
 *
 * Key decisions:
 * - Use the Runtime API because it provides a straightforward way to query properties.
 * - Error handling is performed via a helper function `checkCudaError` to abort on any CUDA API failure.
 * - The program is intentionally simple and portable; it should compile with nvcc on any CUDA-enabled
 *   system.
 *
 * Notes on usage:
 * - Compile with: `nvcc -o check_timeout check_timeout.cu`
 * - Run: `./check_timeout`
 *
 * Edge cases:
 * - No CUDA devices found: prints an informative message.
 * - Devices with incomplete property structures (unlikely) are handled by the error checks.
 *
 * The program focuses solely on the `kernelExecTimeoutEnabled` field but could be extended to report
 * other properties as needed.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// Helper to check CUDA errors
inline void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << msg << " : " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main()
{
    int deviceCount = 0;
    checkCudaError(cudaGetDeviceCount(&deviceCount), "Getting device count");

    if (deviceCount == 0)
    {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        checkCudaError(cudaGetDeviceProperties(&prop, dev), "Getting device properties");

        std::cout << "Device " << dev << ": " << prop.name << std::endl;
        std::cout << "  Kernel Execution Timeout Enabled: "
                  << (prop.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;
    }

    return EXIT_SUCCESS;
}
