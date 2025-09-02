```cpp
/*
Write a program that simply checks if there is at least one CUDA-capable device and prints 'CUDA is available!' or 'No CUDA devices found.'

My thinking process:
1. We need to create a minimal CUDA program that checks for the presence of CUDA-capable devices.
2. The simplest approach is to use the CUDA Runtime API function cudaGetDeviceCount, which returns the number of devices.
3. If the count is greater than 0, we print "CUDA is available!".
4. Otherwise, we print "No CUDA devices found.".
5. The program should include the necessary headers (<cuda_runtime.h> and <iostream>).
6. We will handle the error returned by cudaGetDeviceCount to make sure we don't miss any issues (e.g., if the driver is not installed).
7. The program should compile with nvcc and run on a machine with or without CUDA hardware; it will output the correct message accordingly.
*/

#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "Failed to get device count: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    if (deviceCount > 0) {
        std::cout << "CUDA is available!" << std::endl;
    } else {
        std::cout << "No CUDA devices found." << std::endl;
    }

    return 0;
}
```