```cpp
/*
Query and print the maximum number of threads allowed in a single thread block for the primary CUDA device. This value, `maxThreadsPerBlock`, is a key constraint when designing kernels.

--- Thinking Process ---
To solve this problem we need to query the CUDA driver for the properties of the primary device (device 0). The CUDA Runtime API provides the function `cudaGetDeviceProperties` which fills a `cudaDeviceProp` structure with various device characteristics, including `maxThreadsPerBlock`. The steps are:

1. Include the necessary headers: `<iostream>` for printing and `<cuda_runtime.h>` for CUDA runtime API.
2. Declare a `cudaDeviceProp` variable.
3. Call `cudaGetDeviceProperties(&prop, 0)` to fill it for the primary device. We must check the returned `cudaError_t` to ensure the call succeeded.
4. Extract `prop.maxThreadsPerBlock` and print it to standard output.
5. Return `0` from `main` to indicate successful execution.

No kernel launch is required for this simple query. The program is straightforward and portable across CUDA-capable devices.
*/

#include <iostream>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0); // primary device = 0

    if (err != cudaSuccess) {
        std::cerr << "Error querying device properties: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Maximum number of threads per block (maxThreadsPerBlock): "
              << prop.maxThreadsPerBlock << std::endl;

    return 0;
}
```