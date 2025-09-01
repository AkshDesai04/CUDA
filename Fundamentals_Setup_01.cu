```cu
/*
Aim of the program: 
Write a CUDA program to query and print the number of CUDA-enabled devices available on your system. The program should use the `cudaGetDeviceCount` API call and print a user-friendly message, such as 'Found X CUDA-capable devices.'

Thinking:
1. We need to include the CUDA runtime header `cuda_runtime.h` to use `cudaGetDeviceCount`.
2. Use `int main()` as the entry point.
3. Declare an integer variable to hold the device count.
4. Call `cudaGetDeviceCount(&count)` and capture the returned `cudaError_t`.
5. Check if the call succeeded. If not, print an error message using `cudaGetErrorString(err)` and exit with a non-zero status.
6. If successful, print the message "Found X CUDA-capable devices." using `std::cout`.
7. Return 0 to indicate success.
8. Since we are writing a .cu file, we use `#include <cuda_runtime.h>` and `#include <iostream>`.
9. No kernel launch or other CUDA features are needed; this is purely a host query program.
10. Ensure the code compiles with `nvcc`. 
*/

#include <cuda_runtime.h>
#include <iostream>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Found " << deviceCount << " CUDA-capable devices." << std::endl;
    return 0;
}
```