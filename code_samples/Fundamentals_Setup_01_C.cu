/*
Write a program to explicitly set the active CUDA device to 0 using `cudaSetDevice`. Verify the call was successful by then calling `cudaGetDevice` and printing the result.

My thinking process for this program:
- The aim is straightforward: set the CUDA device to 0, confirm that the device is indeed set by querying it back, and print the result to the console.
- The CUDA runtime API provides `cudaSetDevice(int device)` to set the active device for the current context. The function returns a `cudaError_t` which indicates success or failure. 
- After setting the device, we can retrieve the current active device using `cudaGetDevice(int* device)`, also returning a `cudaError_t`. 
- We need to include error handling: check the return value of each call and, if not `cudaSuccess`, print a helpful error message with `cudaGetErrorString` and exit with a non-zero status. 
- For output, we use standard C++ iostreams (std::cout). 
- The program will be compiled as a CUDA source file (`.cu`) so we must include `<cuda_runtime.h>`. 
- Since the program is very small, a simple `main()` returning 0 on success is sufficient. 
- The code should be minimal, clear, and demonstrate the required functionality: set device, get device, print the device number. 
- Finally, we adhere to the instruction to output only the `.cu` file content, with no additional text. 
*/

#include <iostream>
#include <cuda_runtime.h>

int main()
{
    int device = 0;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    int currentDevice = -1;
    err = cudaGetDevice(&currentDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDevice failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Current active CUDA device: " << currentDevice << std::endl;
    return 0;
}
