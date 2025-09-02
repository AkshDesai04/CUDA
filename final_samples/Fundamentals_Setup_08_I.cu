```cpp
/*
Get the name of the current device and the length of the name string.

Thought Process:
1. The task is to write a CUDA C/C++ program that obtains and displays two pieces of information:
   - The name of the current CUDA device.
   - The length of that name string.

2. In CUDA, device information is retrieved via the runtime API. The relevant function
   `cudaGetDeviceProperties` fills a `cudaDeviceProp` structure that contains a `name`
   member (a null-terminated string of up to 256 characters). The device ID can be
   obtained with `cudaGetDevice`.

3. To determine the string length, we can use `strlen` from `<cstring>`.

4. The program will:
   - Include the necessary headers (`cuda_runtime.h`, `<iostream>`, `<cstring>`).
   - Query the current device ID and its properties.
   - Print the name and its length.
   - Perform basic error checking after each CUDA call.

5. The code is simple and self-contained, making it easy to compile and run on any
   system with CUDA installed.

The resulting file is a valid `.cu` CUDA source file that can be compiled with `nvcc`.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

int main() {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        std::cerr << "Error getting current device: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "Error getting device properties: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Device name: " << prop.name << std::endl;
    std::cout << "Name length: " << std::strlen(prop.name) << std::endl;

    return 0;
}
```