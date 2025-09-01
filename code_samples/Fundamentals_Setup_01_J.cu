```cpp
/*
Query if the primary device has ECC (Error-Correcting Code) memory enabled and print the result.

Thought Process:
1. The goal is to check whether the CUDA primary device (the device currently selected or device 0 if none selected)
   has ECC memory enabled. 
2. CUDA provides the device attribute `cudaDevAttrEccEnabled` which indicates ECC status.
3. Steps:
   - Include necessary headers (`cuda_runtime.h` for CUDA APIs and `<iostream>` for output).
   - Determine the current device using `cudaGetDevice(&device)`. If no device has been set, this will return device 0 by default.
   - Query the ECC attribute using `cudaDeviceGetAttribute(&eccEnabled, cudaDevAttrEccEnabled, device)`.
   - Check the result and print a human‑readable message.
4. Add basic error checking for CUDA calls; if any fail, output an error message and exit.
5. The program is written as a standalone `.cu` file that can be compiled with `nvcc`.
*/

#include <iostream>
#include <cuda_runtime.h>

int main() {
    // Determine the primary device (the one currently selected by the runtime)
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        std::cerr << "Error getting current device: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Query ECC status for this device
    int eccEnabled = 0;
    err = cudaDeviceGetAttribute(&eccEnabled, cudaDevAttrEccEnabled, device);
    if (err != cudaSuccess) {
        std::cerr << "Error querying ECC attribute: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Print result
    if (eccEnabled) {
        std::cout << "ECC is enabled on the primary device (device " << device << ")." << std::endl;
    } else {
        std::cout << "ECC is NOT enabled on the primary device (device " << device << ")." << std::endl;
    }

    return 0;
}
```