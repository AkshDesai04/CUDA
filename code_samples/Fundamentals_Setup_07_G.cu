```cuda
/*
Aim: Print the `major` and `minor` compute capability numbers for the primary device.

Thinking:
- We need to obtain the current primary CUDA device that is active for the host process.
- The CUDA Runtime API provides `cudaGetDevice()` to retrieve the currently active device.
- After getting the device ID, we can query detailed device properties via `cudaGetDeviceProperties()`.
- The `cudaDeviceProp` structure contains `major` and `minor` fields that represent the compute capability.
- We'll handle possible errors: if either API call fails, we should print an error message and exit.
- Finally, print the major and minor values to standard output in a simple format.

The code will be a single .cu file that can be compiled with `nvcc`. It uses the CUDA Runtime API and standard C I/O for output.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get current device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Compute capability of primary device (device %d): %d.%d\n", device, prop.major, prop.minor);
    return 0;
}
```