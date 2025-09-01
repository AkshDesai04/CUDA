/*
Aim of the program:
Query and print whether a kernel execution timeout is enabled on the device.

My thinking process for creating this CUDA program:
1. The CUDA Runtime API provides a structure `cudaDeviceProp` that contains various
   properties of a device. One of those properties is `kernelExecTimeoutEnabled`,
   a boolean that indicates whether the system imposes a limit on how long a
   kernel can run on that device.  When this flag is true, kernels that run longer
   than a certain threshold are terminated (this is mainly for debugging on
   GPUs that are connected to a display, e.g., on laptops).  When it is false,
   kernels can run indefinitely.

2. To query this property we need to:
   - Determine the number of devices available with `cudaGetDeviceCount`.
   - Iterate over each device (or just query device 0 if we only care about one).
   - Use `cudaGetDeviceProperties` to fill a `cudaDeviceProp` structure for that
     device.
   - Read the `kernelExecTimeoutEnabled` member and print the result.

3. The program will be written in standard C++ with CUDA extensions.  The
   `cuda_runtime.h` header provides the necessary API functions.  We'll use
   `std::cout` for printing, so we include `<iostream>`.  A small helper
   function `checkCudaError` will be used to catch any CUDA API errors and
   abort the program with a message.

4. Since the question asks for a .cu file containing the full code, the
   implementation will be a single file that compiles with `nvcc`.  The
   program can be compiled and run on any system with CUDA support.  It
   will output something like:
      Device 0: Kernel execution timeout enabled: Yes
   or
      Device 0: Kernel execution timeout enabled: No

5. Edge cases: If there are no CUDA devices, we should inform the user and
   exit gracefully.  All CUDA calls are checked for errors.

6. The code is kept minimal and clear, with descriptive variable names and
   comments inside the function bodies for readability.

Now, below is the complete .cu file with the multi-line comment at the top,
followed by the actual program code.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <string>

// Helper function to check CUDA API calls
void checkCudaError(cudaError_t err, const std::string& msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main()
{
    int deviceCount = 0;
    checkCudaError(cudaGetDeviceCount(&deviceCount), "Failed to get device count");

    if (deviceCount == 0)
    {
        std::cout << "No CUDA devices found on this system." << std::endl;
        return EXIT_SUCCESS;
    }

    // Query each device
    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        checkCudaError(cudaGetDeviceProperties(&prop, dev), "Failed to get device properties");

        std::cout << "Device " << dev << " (" << prop.name << "): ";
        std::cout << "Kernel execution timeout enabled: "
                  << (prop.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;
    }

    return EXIT_SUCCESS;
}
