```cu
/*
Aim: Based on the asyncEngineCount, print a message like 'Device has X copy engines.' If the count is greater than 0, also print 'Data copy can be overlapped with kernel execution.'

Thinking: 
- The `asyncEngineCount` field in `cudaDeviceProp` represents the number of asynchronous copy engines available on a CUDA device.  
- A count of 0 means the device has no dedicated copy engines; a count > 0 indicates that copy operations can be overlapped with kernel execution.  
- The program will:
  1. Query the number of CUDA devices.
  2. Select the first device (device 0) for simplicity.
  3. Retrieve its properties with `cudaGetDeviceProperties`.
  4. Output the message with the copy engine count.
  5. If the count is greater than zero, also output the overlapping capability message.
- Error checking is performed for each CUDA API call, with error messages printed to `std::cerr` and the program returning a non‑zero exit code on failure.
*/

#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device count: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return -1;
    }

    int device = 0;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Device has " << prop.asyncEngineCount << " copy engines." << std::endl;
    if (prop.asyncEngineCount > 0) {
        std::cout << "Data copy can be overlapped with kernel execution." << std::endl;
    }

    return 0;
}
```