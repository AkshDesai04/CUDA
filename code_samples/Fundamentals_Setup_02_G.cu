/*
For each device, print its memory bus width in bits.

The goal of this program is to query every CUDA-capable device available on the system and
output the width of its memory bus in bits.  This requires using the CUDA runtime API,
specifically the functions cudaGetDeviceCount() to determine how many devices are present
and cudaGetDeviceProperties() to obtain the properties of each device.  The relevant field
in the cudaDeviceProp structure is memoryBusWidth, which reports the bus width in bits.

The program follows these steps:

1. Include the necessary headers: <cstdio> for printf and <cuda_runtime.h> for the CUDA
   runtime API.
2. Call cudaGetDeviceCount() to find the number of devices.  If an error occurs, report
   it and exit with a non‑zero status.
3. Iterate over each device index from 0 to count-1.
4. For each index, call cudaGetDeviceProperties() to retrieve the cudaDeviceProp
   structure.  If this call fails, print an error message and continue to the next device.
5. Print the device index and the memory bus width in bits in a human‑readable format.
6. Exit the program with a zero status.

The program uses C linkage for simplicity and can be compiled with NVCC as follows:
   nvcc -o memory_bus_width memory_bus_width.cu
Then run it:
   ./memory_bus_width
*/

#include <cstdio>
#include <cuda_runtime.h>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0)
    {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to get properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: memory bus width = %d bits\n", dev, prop.memoryBusWidth);
    }

    return 0;
}
