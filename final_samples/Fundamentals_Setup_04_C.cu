/*
Aim of the program:
Query and print the maximum number of threads per multiprocessor for the default device.

Thinking process:
1. Use the CUDA Runtime API to query device properties. The function
   `cudaGetDeviceProperties()` fills a `cudaDeviceProp` structure with
   information about a CUDA device.
2. The default device is device 0, so we call the function with `dev=0`.
3. Within the `cudaDeviceProp` structure, the field `maxThreadsPerMultiProcessor`
   holds the maximum number of threads that can be resident on a single
   multiprocessor for that device.
4. After retrieving the property, simply print the value to the console.
5. Add minimal error checking to ensure the query succeeded.
6. Keep the program self‑contained: include <cuda_runtime.h> and <stdio.h>.
7. The program compiles with `nvcc` and runs on any system with a CUDA capable
   device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int dev = 0;  // Default device
    cudaDeviceProp prop;

    // Query device properties
    cudaError_t err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    // Print the maximum number of threads per multiprocessor
    printf("Maximum number of threads per multiprocessor (device %d): %d\n",
           dev, prop.maxThreadsPerMultiProcessor);

    return 0;
}
