```cpp
/*
Query and print the maximum number of threads per multiprocessor.

Thinking:
The CUDA runtime API provides the structure `cudaDeviceProp` which contains many
device properties. The field `maxThreadsPerMultiProcessor` holds the maximum
number of threads that can be resident on a single multiprocessor (SM) at one
time.  To retrieve this value we need to:
1. Query the current device or choose a device (here we just use device 0).
2. Call `cudaGetDeviceProperties` to fill a `cudaDeviceProp` struct.
3. Access `maxThreadsPerMultiProcessor` and print it.

We'll also check for errors from CUDA API calls and print a useful message if
something fails. The program is self‑contained and can be compiled with
`nvcc -o query_threads_per_mp query_threads_per_mp.cu`.

*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

int main(void)
{
    int device = 0; // We'll use device 0 for simplicity
    cudaDeviceProp prop;

    // Get properties for the chosen device
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    // Print the maximum number of threads per multiprocessor
    printf("Maximum number of threads per multiprocessor on device %d: %d\n",
           device, prop.maxThreadsPerMultiProcessor);

    return 0;
}
```