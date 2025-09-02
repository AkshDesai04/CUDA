```cu
/*
Aim of the program: Try to get device properties for an invalid device index with `cudaGetDeviceProperties`.

My entire thinking:
- The goal is to demonstrate the behavior of the CUDA runtime when `cudaGetDeviceProperties` is called with an index that does not correspond to any physical device.
- First, we query the total number of CUDA-capable devices on the system using `cudaGetDeviceCount`. 
- Then we deliberately choose an invalid index. Two common ways to do this are:
  1. Using a negative index (e.g., -1).
  2. Using an index that is greater than or equal to the number of devices (e.g., `num_devices + 1`).
  For this program, I chose the second approach because it clearly demonstrates that the index is out of bounds.
- The call to `cudaGetDeviceProperties` will return an error code. According to the CUDA Runtime API documentation, the function returns `cudaErrorInvalidDevice` for an out-of-range device index.
- We capture the returned error code, convert it to a human-readable string using `cudaGetErrorString`, and print it. This confirms that the runtime correctly flags the misuse.
- For completeness, the program also prints the number of devices found and the attempted invalid index.
- No device context is set or used; the call to `cudaGetDeviceProperties` does not require a prior `cudaSetDevice` if only retrieving properties.
- The program is written in standard C syntax with CUDA extensions, compiled with `nvcc`, and can be run on any system with the CUDA toolkit installed.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int num_devices = 0;
    cudaError_t err = cudaGetDeviceCount(&num_devices);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Number of CUDA devices on this system: %d\n", num_devices);

    // Intentionally use an invalid device index (num_devices is out-of-range)
    int invalid_index = num_devices; // valid indices are 0 .. num_devices-1
    printf("Attempting to get device properties for invalid index %d...\n", invalid_index);

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, invalid_index);

    if (err != cudaSuccess) {
        printf("cudaGetDeviceProperties returned error: %s (code %d)\n",
               cudaGetErrorString(err), err);
    } else {
        // This block should not execute for an invalid index
        printf("Device %d properties:\n", invalid_index);
        printf("  Name: %s\n", prop.name);
        printf("  Total Global Memory: %lu bytes\n", prop.totalGlobalMem);
        printf("  Multiprocessor Count: %d\n", prop.multiProcessorCount);
    }

    return 0;
}
```