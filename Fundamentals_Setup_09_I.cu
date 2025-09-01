/*
Print the amount of shared memory available per multiprocessor.

To solve this problem we need to query the device properties using CUDA’s
runtime API.  The structure cudaDeviceProp contains a field
sharedMemPerMultiprocessor that holds the amount of shared memory that
each multiprocessor on the GPU can provide.  We will:
1. Call cudaGetDeviceProperties for the first device (device 0).
2. Check for errors from the CUDA API.
3. Print the value in bytes, using the %zu format specifier for size_t.

The program is written in CUDA C and can be compiled with nvcc.  It
requires no kernel launches; all work is done on the host side.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    cudaDeviceProp prop;
    int device = 0; // default to the first CUDA device

    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n",
                cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Shared memory per multiprocessor: %zu bytes\n",
           prop.sharedMemPerMultiprocessor);

    return EXIT_SUCCESS;
}
