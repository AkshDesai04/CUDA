```cpp
/* 
Write a program that queries and prints the total amount of global memory available on device 0 in megabytes (MB). 
You will need to get the `totalGlobalMem` property from `cudaDeviceProp` and convert it from bytes to MB. 

Thought process:
- We need a simple CUDA C program that uses the CUDA Runtime API.
- We'll include <cuda_runtime.h> and <stdio.h>.
- Use cudaGetDeviceProperties to obtain the device properties for device 0.
- Check for errors from the CUDA API call.
- The `totalGlobalMem` field gives the size in bytes; convert to megabytes by dividing by (1024*1024).
- Print the result with printf, formatting as an integer or float.
- Add basic error handling for the CUDA call.
- Compile with nvcc (e.g., nvcc -o query_mem query_mem.cu).
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    cudaDeviceProp prop;
    int device = 0;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error retrieving device properties for device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    double totalMB = (double)prop.totalGlobalMem / (1024.0 * 1024.0);
    printf("Total global memory on device %d: %.2f MB\n", device, totalMB);

    return 0;
}
```