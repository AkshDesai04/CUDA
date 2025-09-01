/*
Write a program that simply checks if there is at least one CUDA-capable device and prints 'CUDA is available!' or 'No CUDA devices found.'

In designing this program I considered the simplest way to query the CUDA runtime for the number of devices. The CUDA runtime API provides cudaGetDeviceCount, which returns the number of devices present. A return value of zero indicates no devices; any positive integer indicates at least one device. I also decided to check the returned cudaError_t to catch any runtime errors that might occur (for example, if the CUDA runtime cannot be initialized). The program uses standard C++ I/O for printing to the console. Since the requirement was to use CUDA with C, I kept the code minimal and focused on the runtime query rather than launching kernels or doing device memory allocations.

Key decisions:
- Use cudaGetDeviceCount to determine the number of CUDA devices.
- Check the returned error code and handle it gracefully.
- Print "CUDA is available!" when count > 0.
- Print "No CUDA devices found." when count == 0.
- The program is written in a single .cu file and can be compiled with nvcc.

Edge cases considered:
- If cudaGetDeviceCount returns an error (e.g., runtime not initialized), we output a generic error message and exit.
- If the device count is zero, we report that no devices were found.
*/

#include <iostream>
#include <cuda_runtime.h>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess)
    {
        std::cerr << "Failed to get CUDA device count: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    if (deviceCount > 0)
    {
        std::cout << "CUDA is available!" << std::endl;
    }
    else
    {
        std::cout << "No CUDA devices found." << std::endl;
    }

    return EXIT_SUCCESS;
}
