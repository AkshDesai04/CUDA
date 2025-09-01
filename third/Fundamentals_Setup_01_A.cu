```cuda
/* 
Aim of the program: Write a program that simply checks if there is at least one CUDA-capable device and prints 'CUDA is available!' or 'No CUDA devices found.'

Thinking:
1. The requirement is to write a minimal CUDA host program that checks for the presence of a CUDA-capable device.
2. The CUDA Runtime API provides cudaGetDeviceCount() which returns the number of devices in 'int *count' and a cudaError_t status.
3. The program should:
   a. Call cudaGetDeviceCount(&count).
   b. If the call fails (non-zero status), we treat that as no device found (or we could print an error).
   c. If count > 0, print "CUDA is available!".
   d. Otherwise print "No CUDA devices found."
4. Use C++ standard output (std::cout) for printing. 
5. Include necessary headers: <iostream> for std::cout, <cuda_runtime.h> for CUDA API.
6. Return 0 on success.
7. The code must compile with nvcc as a .cu file.
8. The entire comment block should contain the exact aim text and the thinking text. No additional output should be present.

Implementation details:
- The program is very small; no kernel launch is needed.
- We'll handle the error code from cudaGetDeviceCount by checking if it's cudaSuccess; otherwise, we can print the error string via cudaGetErrorString.
- Use a simple if-else structure for printing the correct message.

End of thinking. 
*/

#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        // If an error occurs (e.g., driver not available), treat as no device.
        std::cout << "No CUDA devices found." << std::endl;
        return 0;
    }

    if (deviceCount > 0) {
        std::cout << "CUDA is available!" << std::endl;
    } else {
        std::cout << "No CUDA devices found." << std::endl;
    }

    return 0;
}
```