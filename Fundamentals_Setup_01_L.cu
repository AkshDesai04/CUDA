```c
/* 
Aim: Query and print the maximum number of threads that can be active on a single multiprocessor for the primary device.

Thinking:
- The CUDA Runtime API provides a struct `cudaDeviceProp` that contains properties of a device.
- One of the members, `maxThreadsPerMultiProcessor`, gives the maximum number of resident threads that can be active on a single SM (multiprocessor) for that device. This is the theoretical upper bound for a single kernel launch.
- To obtain this value, we first query the primary device (device 0) using `cudaGetDeviceProperties`.
- It is good practice to check for errors after CUDA API calls.
- After retrieving the property, we simply print it to the console.
- This program does not launch any kernels; it only reports the device capability.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    // Use device 0 (primary device)
    int device = 0;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error setting device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    // Get device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties for device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    // Output the maximum number of resident threads per multiprocessor
    printf("Maximum number of resident threads per multiprocessor on device %d: %d\n",
           device, prop.maxThreadsPerMultiProcessor);

    return 0;
}
```