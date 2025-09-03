/*
Get the name of the current device and the length of the name string.

To solve this problem, I need to use the CUDA Runtime API to query information about the device. The steps are straightforward:

1. Include the necessary headers: `<stdio.h>` for input/output and `<cuda_runtime.h>` for CUDA functions.
2. In the `main` function, obtain the current device number using `cudaGetDevice`. This function fills an integer with the device ID currently in use by the calling host thread.
3. Allocate a `cudaDeviceProp` structure and fill it with the device properties using `cudaGetDeviceProperties`. This will give access to the `name` field, which holds the device name as a null-terminated string.
4. Compute the length of the name string using `strlen` from the C standard library.
5. Print the device name and its length to the console.
6. Add basic error checking after each CUDA API call to catch and report failures. If any error occurs, print the CUDA error string using `cudaGetErrorString` and exit with a non-zero status.
7. Compile the file with `nvcc` as a .cu file: e.g., `nvcc -o device_info device_info.cu`.

The program is intentionally minimalistic, focusing only on retrieving and displaying the device name and its length. No advanced CUDA kernel launch or device memory allocation is necessary for this task.

*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <string.h>

int main(void) {
    int device_id;
    cudaError_t err;

    // Get the current device ID
    err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting current device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Retrieve device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Compute length of the device name string
    size_t name_len = strlen(prop.name);

    // Print the results
    printf("Device Name: %s\n", prop.name);
    printf("Length of Device Name: %zu\n", name_len);

    return 0;
}
