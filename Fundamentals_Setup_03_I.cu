/*
Calculate and print the theoretical peak memory bandwidth of the primary device in GB/s. Formula: `(memoryClockRate_kHz * 1000 * memoryBusWidth_bits / 8) / 1e9 * 2` (the *2 is for DDR memory).

Thinking:
1. We need to query CUDA device properties for the primary (device 0). The relevant fields are:
   - memoryClockRate (in kHz)
   - memoryBusWidth (in bits)
2. The theoretical peak bandwidth formula is given. We'll compute it in double precision.
3. We'll print the result in GB/s with a reasonable number of decimal places.
4. Use CUDA runtime API functions: cudaGetDeviceCount, cudaGetDeviceProperties.
5. Include necessary headers: <cuda_runtime.h>, <iostream>.
6. Handle errors gracefully: if no device found, print error and exit.
7. Since the program is simple, no additional functions are needed.

Now we provide the full .cu code below.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0)
    {
        std::cerr << "No CUDA devices found or error querying device count: "
                  << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    // Use the primary device (device 0)
    cudaDeviceProp deviceProp;
    err = cudaGetDeviceProperties(&deviceProp, 0);
    if (err != cudaSuccess)
    {
        std::cerr << "Error getting device properties: "
                  << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    // Retrieve memory clock rate (kHz) and memory bus width (bits)
    unsigned int memoryClockRate_kHz = deviceProp.memoryClockRate;
    unsigned int memoryBusWidth_bits = deviceProp.memoryBusWidth;

    // Compute theoretical peak bandwidth in GB/s
    double bandwidthGBps = (static_cast<double>(memoryClockRate_kHz)
                            * 1000.0      // convert kHz to Hz
                            * static_cast<double>(memoryBusWidth_bits) / 8.0) // bytes per second
                           / 1e9                     // convert to GB/s
                           * 2.0;                    // DDR multiplier

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Primary CUDA device: " << deviceProp.name << std::endl;
    std::cout << "Memory clock rate: " << memoryClockRate_kHz << " kHz" << std::endl;
    std::cout << "Memory bus width: " << memoryBusWidth_bits << " bits" << std::endl;
    std::cout << "Theoretical peak memory bandwidth: " << bandwidthGBps
              << " GB/s" << std::endl;

    return EXIT_SUCCESS;
}
