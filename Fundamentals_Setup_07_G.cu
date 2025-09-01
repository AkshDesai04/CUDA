```cpp
/*
Print the `major` and `minor` compute capability numbers for the primary device.
This program demonstrates how to query the compute capability of the default
CUDA device (device 0).  The steps are:
1. Query the number of CUDA devices.
2. Select device 0 as the primary device.
3. Retrieve its properties using cudaGetDeviceProperties.
4. Extract the major and minor compute capability numbers.
5. Print them to standard output.

The program uses the CUDA Runtime API. It performs minimal error checking
to ensure that the device properties are obtained successfully.  The output
is formatted as:
    Major: <major>, Minor: <minor>
*/

#include <iostream>
#include <cuda_runtime.h>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to get device count: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (deviceCount == 0)
    {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }

    int device = 0; // primary device
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Major: " << prop.major << ", Minor: " << prop.minor << std::endl;
    return 0;
}
```