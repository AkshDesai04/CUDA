```cuda
/* 
Aim of the program: Write a function `float get_clock_rate_ghz(int device_id)` that returns the clock rate in GHz for a given device. 

Thinking: 
- The CUDA Runtime API provides `cudaDeviceProp` structure which contains the field `clockRate` expressed in kilohertz (kHz). 
- To obtain the clock rate for a particular device, we need to call `cudaGetDeviceProperties(&prop, device_id)`. 
- After retrieving the properties, we can convert the clock rate from kHz to gigahertz (GHz) by multiplying by 1e-6 (since 1 kHz = 1e-6 GHz). 
- The function should return a `float`. If an error occurs (e.g., invalid device id), it should return a negative value to indicate failure. 
- No external dependencies beyond the CUDA Runtime API are required. 
- The implementation will be placed in a .cu file so it can be compiled with nvcc. 
*/

#include <cuda_runtime.h>
#include <stdio.h>

/**
 * @brief Returns the clock rate of the specified CUDA device in GHz.
 *
 * @param device_id The ID of the CUDA device.
 * @return float Clock rate in GHz. Returns -1.0f on error.
 */
float get_clock_rate_ghz(int device_id)
{
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);

    if (err != cudaSuccess)
    {
        // Optional: print error message
        // fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n", device_id, cudaGetErrorString(err));
        return -1.0f;
    }

    // prop.clockRate is in kilohertz; convert to gigahertz.
    return prop.clockRate * 1e-6f;
}
```