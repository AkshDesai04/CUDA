```cuda
/*
Aim of the program:
Query and print the maximum texture dimension for a 1D texture using cudaDeviceGetAttribute with cudaDevAttrMaxTexture1DWidth.

Thinking:
- The goal is to obtain the device attribute that specifies the maximum width for a 1D texture.
- We'll use cudaDeviceGetAttribute, which requires the attribute enum, a pointer to an int to receive the value, and the device number.
- For simplicity, we target device 0 (the default CUDA device). If needed, cudaSetDevice(0) can be called before querying.
- We'll check the return status of the CUDA API calls. If an error occurs, we print an error message and exit.
- Finally, we output the maximum width using std::cout.
- The program is a minimal CUDA host code and does not launch any kernels.
*/

#include <cuda_runtime.h>
#include <iostream>

int main()
{
    int device = 0;                      // Use the default device
    int maxTexture1DWidth = 0;           // Variable to store the attribute

    // Optionally set the device (not strictly necessary for device 0)
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Query the maximum 1D texture width for the selected device
    err = cudaDeviceGetAttribute(&maxTexture1DWidth,
                                 cudaDevAttrMaxTexture1DWidth,
                                 device);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaDeviceGetAttribute failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Print the result
    std::cout << "Maximum 1D texture width for device " << device
              << " is " << maxTexture1DWidth << std::endl;

    return 0;
}
```