```cuda
/*
Aim of the program:
Write a function `float get_clock_rate_ghz(int device_id)` that returns the clock rate in GHz for a given device.

Thinking:
- The CUDA runtime API provides `cudaGetDeviceProperties()` which fills a `cudaDeviceProp` structure.
- The field `clockRate` in that structure is expressed in kilohertz (kHz).
- To convert kHz to gigahertz (GHz) we divide by 1,000,000 (since 1 GHz = 1,000,000 kHz).
- The function will first set the desired device with `cudaSetDevice(device_id)`.
- It will then retrieve the device properties, perform the conversion, and return the result.
- If any CUDA call fails (e.g., invalid device id), the function will return -1.0f to signal an error.
- A small `main` is provided for demonstration purposes; it queries all available devices and prints their clock rates.
- Compile with `nvcc` and run to see the results.

*/

#include <stdio.h>
#include <cuda_runtime.h>

/**
 * @brief Retrieves the clock rate of a CUDA device in GHz.
 *
 * @param device_id The ID of the CUDA device.
 * @return float The clock rate in GHz, or -1.0f on error.
 */
float get_clock_rate_ghz(int device_id) {
    // First, validate the device id by attempting to set it.
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        // Invalid device ID or other error.
        return -1.0f;
    }

    // Retrieve device properties.
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        // Failed to get properties.
        return -1.0f;
    }

    // Convert clock rate from kHz to GHz: 1 GHz = 1,000,000 kHz.
    float clock_rate_ghz = static_cast<float>(prop.clockRate) / 1e6f;
    return clock_rate_ghz;
}

int main(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (device_count == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    printf("Detected %d CUDA device(s):\n", device_count);
    for (int i = 0; i < device_count; ++i) {
        float ghz = get_clock_rate_ghz(i);
        if (ghz < 0.0f) {
            printf("  Device %d: Error retrieving clock rate.\n", i);
        } else {
            printf("  Device %d: Clock rate = %.3f GHz\n", i, ghz);
        }
    }

    return 0;
}
```