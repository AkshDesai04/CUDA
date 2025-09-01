```cpp
/*
 Aim of the program:
 Write a function `get_gpu_memory_mb(int device_id)` that returns the total global memory in MB for the given device.

 My approach:
 1. Use the CUDA Runtime API because it provides a straightforward way to query device properties.
 2. To obtain the total global memory, the `cudaDeviceProp` structure offers the field `totalGlobalMem` which is in bytes.
 3. Convert bytes to megabytes by dividing by (1024 * 1024).
 4. Handle errors by checking the return values of CUDA API calls and printing descriptive messages if something fails.
 5. Ensure the device is set before querying its properties using `cudaSetDevice`.
 6. The function returns an `int` representing megabytes. If an error occurs, return -1 as an indicator.
 7. Provide a simple `main` function to demonstrate usage, though it's not strictly required by the prompt.

 This results in a self‑contained `.cu` file that compiles with `nvcc` and can be executed to query GPU memory.
*/
#include <cuda_runtime.h>
#include <stdio.h>

/**
 * @brief Returns the total global memory of the specified CUDA device in megabytes.
 *
 * @param device_id The zero‑based index of the CUDA device.
 * @return int Total global memory in MB on success, -1 on failure.
 */
int get_gpu_memory_mb(int device_id)
{
    // Ensure the device index is valid
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (device_id < 0 || device_id >= deviceCount)
    {
        fprintf(stderr, "Invalid device_id %d. Available devices: 0 to %d\n", device_id, deviceCount - 1);
        return -1;
    }

    // Set the current device
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", device_id, cudaGetErrorString(err));
        return -1;
    }

    // Retrieve device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n", device_id, cudaGetErrorString(err));
        return -1;
    }

    // Convert bytes to megabytes
    const size_t bytesPerMB = 1024ULL * 1024ULL;
    int mem_mb = static_cast<int>(prop.totalGlobalMem / bytesPerMB);
    return mem_mb;
}

/* Optional demonstration of the function */
int main()
{
    int device_id = 0; // Query the first device by default
    int mem_mb = get_gpu_memory_mb(device_id);
    if (mem_mb >= 0)
    {
        printf("Device %d has %d MB of global memory.\n", device_id, mem_mb);
    }
    else
    {
        printf("Failed to query memory for device %d.\n", device_id);
    }
    return 0;
}
```