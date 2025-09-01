/*
Aim: Query and print the maximum texture dimension for a 1D texture using `cudaDeviceGetAttribute` with `cudaDevAttrMaxTexture1DWidth`.

Thinking:
The task requires a CUDA C program that determines the largest width that a 1D texture can have on the current device. The CUDA Runtime API provides `cudaDeviceGetAttribute` to query various device attributes, including `cudaDevAttrMaxTexture1DWidth`. The function signature is:
    cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device);
Here, `value` will store the maximum width, `attr` is the enum `cudaDevAttrMaxTexture1DWidth`, and `device` specifies which GPU device to query. For simplicity, we will query device 0, which is the default active device. Error handling is essential: we will check the returned `cudaError_t` and print an informative message if something goes wrong. The program will print the maximum width in a human‑readable form. Since the attribute is an `int`, we will use an `int` variable to hold the result. The program is a single file suitable for compiling with `nvcc`. No additional libraries are required beyond the CUDA Runtime and standard C I/O. The program will be named `query_max_texture1d.cu`.

Edge cases:
- On devices that do not support 1D textures, the attribute might be zero or a negative value. We will simply print whatever is returned.
- If the device is not set or cudaGetDevice fails, we assume device 0.

The final code follows the standard structure: include headers, main function, query, error check, print, and exit. No extraneous output is produced.*/

// query_max_texture1d.cu
#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int max1dWidth = 0;
    int device = 0;  // Query the default device (0)

    // Ensure we have a valid device context
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get current device: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Query the maximum 1D texture width
    err = cudaDeviceGetAttribute(&max1dWidth, cudaDevAttrMaxTexture1DWidth, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetAttribute failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Print the result
    printf("Maximum 1D texture width on device %d: %d\n", device, max1dWidth);

    return EXIT_SUCCESS;
}
