/*
Based on the `asyncEngineCount`, print a message like 'Device has X copy engines.' If the count is greater than 0, also print 'Data copy can be overlapped with kernel execution.'
I first considered that the requirement is to use CUDA runtime API to query device properties. The key field is asyncEngineCount in cudaDeviceProp. The program will query the number of devices, then for each device fetch properties, and print the desired messages. Since the prompt asks for a simple program, I used printf for output. Error checking is minimal but sufficient to demonstrate proper API usage. I chose to print messages for the first device (device 0) for simplicity, but could be extended to all devices if desired. The code is self-contained and compiled as a .cu file. It includes the necessary headers, sets the device, and prints the required messages. No additional libraries beyond the CUDA runtime are needed. 
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        printf("Failed to get device count: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0)
    {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    // For simplicity, use the first device
    int dev = 0;
    err = cudaSetDevice(dev);
    if (err != cudaSuccess)
    {
        printf("Failed to set device %d: %s\n", dev, cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess)
    {
        printf("Failed to get properties for device %d: %s\n", dev, cudaGetErrorString(err));
        return -1;
    }

    printf("Device has %d copy engines.\n", prop.asyncEngineCount);
    if (prop.asyncEngineCount > 0)
    {
        printf("Data copy can be overlapped with kernel execution.\n");
    }

    return 0;
}
