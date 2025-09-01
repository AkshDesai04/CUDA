/*
Aim: Query and print the maximum dimensions (x, y, z) of a thread block for the primary CUDA device. The result should be displayed clearly, for example: 'Max Block Dimensions: [X, Y, Z]'.

Thinking:
- The maximum block dimensions are stored in the cudaDeviceProp structure returned by cudaGetDeviceProperties. The fields of interest are maxThreadsDim.x, maxThreadsDim.y, and maxThreadsDim.z.
- We'll query the properties of the primary device (device 0) using cudaGetDeviceProperties.
- We'll perform error checking on the CUDA API call and exit gracefully if an error occurs.
- Finally, we'll print the dimensions in the requested format using std::cout.
- No kernel is required; this is purely a host-side query.
- The code is written in CUDA C/C++ and can be compiled with nvcc.
*/

#include <iostream>
#include <cuda_runtime.h>

int main() {
    int device = 0; // primary device
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Max Block Dimensions: [" 
              << prop.maxThreadsDim.x << ", " 
              << prop.maxThreadsDim.y << ", " 
              << prop.maxThreadsDim.z << "]" << std::endl;

    return EXIT_SUCCESS;
}
