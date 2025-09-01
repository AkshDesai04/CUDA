/*
Aim: Write a program to explicitly set the active CUDA device to 0 using `cudaSetDevice`. Verify the call was successful by then calling `cudaGetDevice` and printing the result.

Thinking: 
- Include <cuda_runtime.h> to access CUDA runtime API functions.
- Use `cudaSetDevice(0)` to set the active device to GPU 0.
- Check the returned `cudaError_t`; if it's not `cudaSuccess`, print the error string via `cudaGetErrorString` and exit.
- After setting, call `cudaGetDevice(&dev)` to retrieve the currently active device index.
- Print the index using `printf`.
- For good practice, reset the device with `cudaDeviceReset()` before exiting, though it is optional for this simple demo.
- Compile with `nvcc` (e.g., `nvcc -o set_device_demo set_device_demo.cu`).
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    // Explicitly set the active CUDA device to 0
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Retrieve the currently active device
    int deviceId = -1;
    err = cudaGetDevice(&deviceId);
    if (err != cudaSuccess) {
        printf("cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Print the active device index
    printf("Active CUDA device is %d\n", deviceId);

    // Optional: Reset the device before exiting
    cudaDeviceReset();

    return 0;
}
