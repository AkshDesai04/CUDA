/*
Print the total constant memory on the device in bytes.
This program queries the CUDA runtime for the properties of the first CUDA device
(assuming there is at least one device present). It retrieves the `totalConstMem`
field from the `cudaDeviceProp` structure, which represents the total amount
of constant memory available on the device in bytes. The value is then printed
to standard output. If the device cannot be queried or if there is no CUDA
device, the program will report an error. This simple example demonstrates
how to access device properties using the CUDA runtime API in C/C++.
*/
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0)
    {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0); // query the first device
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Total constant memory on device 0: %zu bytes\n", prop.totalConstMem);

    return EXIT_SUCCESS;
}
