```cu
/*
 * Aim of the program: Write a function `float get_clock_rate_ghz(int device_id)` that returns the clock rate in GHz for a given device.
 *
 * Thinking:
 * - To obtain the clock rate of a CUDA device, we use the CUDA Runtime API `cudaGetDeviceProperties`.
 * - The `cudaDeviceProp` structure contains a member `clockRate` which reports the device clock rate in kilohertz (kHz).
 * - To convert this to gigahertz (GHz), we divide by 1,000,000 (since 1 GHz = 1,000,000 kHz).
 * - The function should handle invalid device IDs or errors from `cudaGetDeviceProperties`. In case of an error, we return 0.0f.
 * - We will include error checking by inspecting the return status of `cudaGetDeviceProperties`.
 * - For demonstration, a simple `main` function is provided that prints the clock rate of the first device.
 * - The program uses standard CUDA includes (`cuda_runtime.h`) and error handling macros.
 * - The code is self‑contained in a single .cu file.
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Function to get the clock rate of a CUDA device in GHz */
float get_clock_rate_ghz(int device_id) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        // Handle error: print message and return 0.0
        fprintf(stderr, "Error: cudaGetDeviceProperties failed for device %d: %s\n",
                device_id, cudaGetErrorString(err));
        return 0.0f;
    }

    /* prop.clockRate is in kilohertz (kHz). Convert to gigahertz (GHz). */
    float clock_rate_ghz = prop.clockRate / 1.0e6f; // divide by 1,000,000
    return clock_rate_ghz;
}

/* Optional main function for testing */
int main(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDeviceCount failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    if (device_count == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    for (int i = 0; i < device_count; ++i) {
        float ghz = get_clock_rate_ghz(i);
        printf("Device %d clock rate: %.3f GHz\n", i, ghz);
    }

    return 0;
}
```