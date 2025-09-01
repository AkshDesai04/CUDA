```cpp
/*
Aim: Query and print the amount of shared memory available per thread block in bytes for the primary device.

Thinking: 
1. We need to query the CUDA device properties for the primary device (device ID 0 by default).
2. The relevant field in cudaDeviceProp is sharedMemPerBlock which stores the amount of shared memory available per block in bytes.
3. We'll use cudaGetDeviceProperties to fill a cudaDeviceProp struct.
4. After fetching the properties, we print the sharedMemPerBlock value to standard output.
5. We'll include basic error checking: if cudaGetDeviceProperties fails, we print an error and exit.
6. The program is a simple host-only C++ program that uses the CUDA Runtime API, so no kernels are launched.
7. Since we want a .cu file, we include <cuda_runtime.h> and <iostream>.
8. The program will compile with nvcc and run on systems with CUDA installed.
9. We'll keep the code minimal and self-contained, with a single main function.
*/

#include <cuda_runtime.h>
#include <iostream>

int main() {
    int device_id = 0; // primary device
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        std::cerr << "Error getting device properties: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Shared memory per block for device " << device_id
              << ": " << prop.sharedMemPerBlock << " bytes" << std::endl;

    return 0;
}
```