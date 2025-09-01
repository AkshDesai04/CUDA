```cpp
/*
Aim of the program:
Write a function float get_clock_rate_ghz(int device_id) that returns the clock rate in GHz for a given device.

Thinking:
To obtain the clock rate of a CUDA device we use the CUDA Runtime API.  
The function cudaGetDeviceProperties retrieves a struct cudaDeviceProp which contains a field
`clockRate` that specifies the core clock frequency in kilohertz (kHz).  
We simply query the properties for the requested device ID, check for errors, and then
convert kHz to gigahertz (GHz) by dividing by 1,000,000.  

We also provide a small main function that demonstrates how to use the function:
it queries the number of devices, iterates over them, and prints each device’s
clock rate in GHz.  Error handling prints an error message and returns -1.0f
when the device properties cannot be retrieved.
*/

#include <cuda_runtime.h>
#include <stdio.h>

float get_clock_rate_ghz(int device_id)
{
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n",
                device_id, cudaGetErrorString(err));
        return -1.0f;
    }

    // prop.clockRate is in kHz; convert to GHz
    float ghz = prop.clockRate / 1.0e6f;
    return ghz;
}

int main(void)
{
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (device_count == 0)
    {
        printf("No CUDA devices found.\n");
        return 0;
    }

    printf("Number of CUDA devices: %d\n", device_count);

    for (int i = 0; i < device_count; ++i)
    {
        float ghz = get_clock_rate_ghz(i);
        if (ghz >= 0.0f)
            printf("Device %d clock rate: %.3f GHz\n", i, ghz);
        else
            printf("Device %d clock rate: error retrieving device properties\n", i);
    }

    return 0;
}
```