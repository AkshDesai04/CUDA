```cu
/*
Write a conditional print statement: if `concurrentKernels` is true, print 'Device supports concurrency.', otherwise print 'Device does not support concurrency.'

Thinking:
1. To determine if the device supports concurrent kernels, we need to query the device properties via cudaGetDeviceProperties.
2. The cudaDeviceProp structure contains a field called concurrentKernels which is non-zero if the device supports concurrent execution of kernels.
3. We'll write a simple CUDA host program that:
   - Initializes the device (defaults to device 0).
   - Retrieves the device properties.
   - Checks the concurrentKernels flag.
   - Prints the appropriate message using std::cout.
4. We'll include the necessary headers: <cuda_runtime.h> for CUDA API and <iostream> for printing.
5. The program will return 0 on success. No CUDA kernel launch is needed; this is purely a host query program.
6. The output file will be a complete .cu file that can be compiled with nvcc.
*/

#include <cuda_runtime.h>
#include <iostream>

int main()
{
    int device = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (prop.concurrentKernels) {
        std::cout << "Device supports concurrency." << std::endl;
    } else {
        std::cout << "Device does not support concurrency." << std::endl;
    }

    return 0;
}
```