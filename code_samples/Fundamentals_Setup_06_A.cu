```cuda
/*
Aim of the program:
From the maximum dimensions [X, Y, Z], verify that X * Y * Z is not necessarily equal to `maxThreadsPerBlock`
(since X, Y, and Z are individual dimension limits).

Thinking:
- The CUDA driver/API provides per-device properties via `cudaDeviceProp`. Among these properties are:
  * `maxThreadsDim` – an array of three integers giving the maximum dimensions for a kernel launch
    in each dimension (x, y, z). These limits are independent; the product of them can exceed the
    limit imposed on a single block.
  * `maxThreadsPerBlock` – the maximum number of threads that can be launched in one block.
- The product `maxThreadsDim.x * maxThreadsDim.y * maxThreadsDim.z` is the maximum number of threads
  that could be requested if we were to use the maximum allowed size in each dimension. However,
  CUDA restricts the total threads in a block to `maxThreadsPerBlock`, so the product may be
  larger than this limit. This program will query the properties for every CUDA-capable device
  present, compute both values, and print them side‑by‑side to illustrate that they are not
  necessarily equal.
- The program is written in CUDA C++. It includes error checking for CUDA calls and prints a
  clear message indicating whether the product of the max dimensions exceeds `maxThreadsPerBlock`.
- No kernel launch is required – the purpose is purely to query and display device limits.

Implementation steps:
1. Include necessary headers (`<iostream>`, `<cuda_runtime.h>`).
2. Define a helper function `checkCudaError` to wrap CUDA API calls.
3. In `main`:
   a. Call `cudaGetDeviceCount` to find how many devices exist.
   b. For each device:
      - Set the device with `cudaSetDevice`.
      - Retrieve properties with `cudaGetDeviceProperties`.
      - Extract `maxThreadsDim[0..2]` and `maxThreadsPerBlock`.
      - Compute `product = dimX * dimY * dimZ`.
      - Print the device index, dimensions, `maxThreadsPerBlock`, product, and a comparison
        indicating whether the product is larger than the per‑block limit.
4. Compile with `nvcc` and run; the output demonstrates the desired relationship.

The code below follows this plan and is fully self‑contained. It will compile to a single `.cu` file
that, when executed, reports the requested verification for all detected CUDA devices.
*/

#include <iostream>
#include <cuda_runtime.h>

// Helper macro for error checking
inline void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << "Error: " << msg << " (" << cudaGetErrorString(err) << ")\n";
        exit(EXIT_FAILURE);
    }
}

int main()
{
    int deviceCount = 0;
    checkCudaError(cudaGetDeviceCount(&deviceCount), "Getting device count");

    if (deviceCount == 0)
    {
        std::cout << "No CUDA-capable devices found.\n";
        return 0;
    }

    std::cout << "Found " << deviceCount << " CUDA-capable device(s).\n\n";

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        // Set current device
        checkCudaError(cudaSetDevice(dev), "Setting device");

        // Retrieve device properties
        cudaDeviceProp prop;
        checkCudaError(cudaGetDeviceProperties(&prop, dev), "Getting device properties");

        // Extract maximum thread dimensions
        int dimX = prop.maxThreadsDim[0];
        int dimY = prop.maxThreadsDim[1];
        int dimZ = prop.maxThreadsDim[2];

        // Compute product of maximum dimensions
        size_t productDims = static_cast<size_t>(dimX) * dimY * dimZ;

        // Print the results
        std::cout << "Device " << dev << ": \"" << prop.name << "\"\n";
        std::cout << "  maxThreadsDim:   X = " << dimX << ", Y = " << dimY << ", Z = " << dimZ << "\n";
        std::cout << "  maxThreadsPerBlock: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Product of dims   : " << productDims << "\n";

        // Verify relationship
        if (productDims > static_cast<size_t>(prop.maxThreadsPerBlock))
        {
            std::cout << "  Note: The product of the maximum dimensions (" << productDims
                      << ") exceeds maxThreadsPerBlock (" << prop.maxThreadsPerBlock
                      << "). Hence they are NOT necessarily equal.\n";
        }
        else
        {
            std::cout << "  Note: The product of the maximum dimensions (" << productDims
                      << ") is less than or equal to maxThreadsPerBlock (" << prop.maxThreadsPerBlock
                      << ").\n";
        }

        std::cout << std::endl;
    }

    return 0;
}
```