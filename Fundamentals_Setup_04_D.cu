```cpp
/*
Aim of the program:
Calculate and print the theoretical maximum number of concurrent threads the entire GPU can run (`maxThreadsPerMultiProcessor` * `multiProcessorCount`).

Thinking:
- Use CUDA Runtime API to query all CUDA-capable devices on the system.
- For each device, retrieve `cudaDeviceProp` via `cudaGetDeviceProperties`.
- The property `maxThreadsPerMultiProcessor` gives the maximum number of concurrent threads that a single SM can run.
- The property `multiProcessorCount` gives the number of SMs on that device.
- The product of these two values is the maximum number of concurrent threads that the whole device can execute at once.
- Sum this product over all devices to get the theoretical maximum across the entire system.
- Handle errors returned by CUDA API calls and print a helpful message if something goes wrong.
- Finally, print the total theoretical maximum number of concurrent threads.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving device count: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    long long totalConcurrentThreads = 0;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error retrieving properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        long long deviceThreads = (long long)prop.maxThreadsPerMultiProcessor * (long long)prop.multiProcessorCount;
        totalConcurrentThreads += deviceThreads;

        printf("Device %d (%s): maxThreadsPerSM=%d, SMs=%d, concurrent threads=%lld\n",
               dev, prop.name, prop.maxThreadsPerMultiProcessor, prop.multiProcessorCount, deviceThreads);
    }

    printf("\nTheoretical maximum number of concurrent threads the entire GPU(s) can run: %lld\n",
           totalConcurrentThreads);

    return 0;
}
```