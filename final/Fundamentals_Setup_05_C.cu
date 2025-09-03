```cpp
/* 
If a kernel uses 64 registers per thread and the max threads per block is 1024, calculate the total registers required by a full block. Compare this to the max registers per block to see if such a launch is possible.
The program queries the current device properties using cudaGetDeviceProperties to obtain the maximum threads per block and the maximum registers per block supported by the GPU. 
We then compute the registers required by a full block as:
    total_regs_needed = regs_per_thread * max_threads_per_block
where regs_per_thread is fixed at 64 and max_threads_per_block comes from the device properties (normally 1024). 
The maximum registers per block are also retrieved from the device properties. 
Finally, the program prints the total registers needed, the device's limit, and whether the launch configuration is feasible (i.e., total_regs_needed <= max_regs_per_block).
*/

#include <cuda_runtime.h>
#include <iostream>

int main()
{
    int device = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDevice(device);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to set device: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    const int regs_per_thread = 64;
    const int max_threads_per_block = prop.maxThreadsPerBlock; // usually 1024
    const int max_regs_per_block = prop.maxRegsPerBlock;        // device limit

    // Compute total registers needed by a full block
    int total_regs_needed = regs_per_thread * max_threads_per_block;

    std::cout << "Registers per thread: " << regs_per_thread << std::endl;
    std::cout << "Max threads per block (device): " << max_threads_per_block << std::endl;
    std::cout << "Total registers needed for a full block: " << total_regs_needed << std::endl;
    std::cout << "Maximum registers per block (device limit): " << max_regs_per_block << std::endl;

    if (total_regs_needed <= max_regs_per_block)
    {
        std::cout << "Launch is possible: total registers needed do not exceed the device limit." << std::endl;
    }
    else
    {
        std::cout << "Launch NOT possible: total registers needed exceed the device limit." << std::endl;
    }

    return 0;
}
```