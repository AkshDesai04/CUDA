```cuda
/*
Query and print the maximum number of threads that can be active on a single multiprocessor for the primary device.

Thinking process:
- The CUDA runtime API provides cudaGetDeviceProperties to retrieve a cudaDeviceProp struct containing device-specific information.
- The struct has a field called maxThreadsPerMultiProcessor which holds the maximum number of threads that can be resident on any single SM.
- The "primary device" is device index 0 in CUDA's zero-based indexing.
- We'll call cudaGetDeviceProperties with dev=0, handle any errors, and then print the value.
- A minimal, self-contained program will include <stdio.h> and <cuda_runtime.h>, set up the main function, query the properties, and output the result.
- No kernel launch is needed; we only query runtime data.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    cudaDeviceProp prop;
    int device = 0;  // primary device

    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get device properties (error %s)!\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Maximum threads per multiprocessor on device %d (%s): %d\n",
           device, prop.name, prop.maxThreadsPerMultiProcessor);

    return EXIT_SUCCESS;
}
```