/*
Query and print the texture alignment requirement for device 0.
The aim of the program is to retrieve the texture alignment property from
the CUDA device properties for device 0 and print it to the standard
output.  To achieve this, we use the CUDA runtime API function
cudaGetDeviceProperties which fills a cudaDeviceProp structure with
device-specific information.  The texture alignment requirement is
available in the `textureAlignment` field of this structure.  We also
include basic error handling by checking the returned cudaError_t from
the API calls.  The program sets the active device to 0 explicitly,
though CUDA defaults to device 0 if no device is set.  The alignment
value is printed as a size_t integer.  This simple console
application demonstrates how to query device capabilities in a
CUDA-enabled C program.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    cudaError_t err;
    int device = 0;
    cudaDeviceProp prop;

    /* Optionally set the device explicitly */
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error setting device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    /* Retrieve device properties */
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties for device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    /* Print the texture alignment requirement */
    printf("Device %d texture alignment requirement: %zu bytes\n", device, prop.textureAlignment);

    return 0;
}
