```cpp
/*
Query and print the total amount of constant memory available on device 0 in kilobytes (KB).

Thinking:
- The task requires a CUDA C program (.cu file) that outputs the constant memory size of device 0.
- Constant memory size is provided by the CUDA runtime in the cudaDeviceProp struct as totalConstMem (in bytes).
- We need to query device properties for device 0 using cudaGetDeviceProperties.
- Convert bytes to kilobytes by dividing by 1024.
- Output the result using standard output (std::cout).
- Include necessary headers: <cuda_runtime.h> for CUDA functions, <iostream> for printing.
- Perform minimal error checking: if cudaGetDeviceProperties fails, print an error and exit.
- The program will be self-contained: a main function that performs the query and prints the value.
- Since the requirement is just to print the value, no other logic is needed.
- We'll use C++ for simplicity, but C style is also fine. Using std::cout keeps the code clean.
*/

#include <cuda_runtime.h>
#include <iostream>

int main() {
    int device_id = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties for device " << device_id
                  << ": " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    size_t total_const_mem_bytes = prop.totalConstMem;
    size_t total_const_mem_kb = total_const_mem_bytes / 1024;

    std::cout << "Total constant memory on device " << device_id
              << ": " << total_const_mem_kb << " KB" << std::endl;

    return 0;
}
```