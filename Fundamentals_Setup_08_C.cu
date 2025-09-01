```cuda
/*
Aim of the program: Write a program that compares the driver and runtime versions and prints a warning if the driver version is older than the runtime version, as this is an invalid configuration.

Thinking process:
1. The CUDA Driver API provides the function cuDriverGetVersion() which returns an integer where the upper 16 bits represent the major version and the lower 16 bits represent the minor version (e.g., 11020 for CUDA 11.2).  
2. The CUDA Runtime API provides cudaRuntimeGetVersion() with the same encoding.
3. To compare the two versions we simply compare the integer values: a lower integer value means the driver is older.
4. We need to initialize the Driver API with cuInit(0) before calling cuDriverGetVersion.  
5. Error handling: If any CUDA API call fails we print an error message and exit with a non‑zero status.  
6. If the driver version is older than the runtime version, we print a warning message.  
7. The program is a simple command‑line executable that prints the versions and the warning if applicable.  
8. Include headers <cuda.h> for the Driver API and <cuda_runtime.h> for the Runtime API.  
9. Use std::cout for output, and return 0 on success.

*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

int main()
{
    // Initialize the Driver API
    CUresult cuRes = cuInit(0);
    if (cuRes != CUDA_SUCCESS) {
        std::cerr << "cuInit failed: " << cuRes << std::endl;
        return EXIT_FAILURE;
    }

    // Get driver version
    int driverVer = 0;
    cuRes = cuDriverGetVersion(&driverVer);
    if (cuRes != CUDA_SUCCESS) {
        std::cerr << "cuDriverGetVersion failed: " << cuRes << std::endl;
        return EXIT_FAILURE;
    }

    // Get runtime version
    int runtimeVer = 0;
    cudaError_t rtRes = cudaRuntimeGetVersion(&runtimeVer);
    if (rtRes != cudaSuccess) {
        std::cerr << "cudaRuntimeGetVersion failed: " << rtRes << std::endl;
        return EXIT_FAILURE;
    }

    // Print versions
    std::cout << "Driver Version  : " << (driverVer / 1000) << "."
              << ((driverVer % 1000) / 10) << std::endl;
    std::cout << "Runtime Version : " << (runtimeVer / 1000) << "."
              << ((runtimeVer % 1000) / 10) << std::endl;

    // Compare versions
    if (driverVer < runtimeVer) {
        std::cerr << "WARNING: The driver version (" << (driverVer / 1000) << "."
                  << ((driverVer % 1000) / 10) << ") is older than the runtime version ("
                  << (runtimeVer / 1000) << "." << ((runtimeVer % 1000) / 10)
                  << "). This is an invalid configuration." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```