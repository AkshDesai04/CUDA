/*
Aim of the program:
Calculate and print the theoretical maximum number of concurrent threads the entire GPU can run (`maxThreadsPerMultiProcessor` * `multiProcessorCount`).

Thought process:
- To determine the theoretical maximum number of concurrent threads that the GPU can run, we need the number of multiprocessors (SMs) and the maximum number of threads that can run on each SM.
- CUDA provides this information in the `cudaDeviceProp` structure obtained via `cudaGetDeviceProperties`.
- The program will:
  1. Query the number of CUDA-capable devices available.
  2. If at least one device exists, it will retrieve the properties of the first device (device 0).
  3. Extract `multiProcessorCount` and `maxThreadsPerMultiProcessor`.
  4. Multiply them to get the total theoretical concurrent threads.
  5. Print the result to standard output.
- Error handling is included to catch failures in querying device properties.
- The program is written in C and compiled as a CUDA `.cu` file. It uses the standard CUDA runtime API.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    // Use device 0 for this calculation
    int device = 0;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    int multiProcessorCount = prop.multiProcessorCount;

    long long theoreticalMaxThreads = (long long)maxThreadsPerSM * (long long)multiProcessorCount;

    printf("Device %d: %s\n", device, prop.name);
    printf("Maximum threads per SM: %d\n", maxThreadsPerSM);
    printf("Number of SMs: %d\n", multiProcessorCount);
    printf("Theoretical maximum number of concurrent threads the entire GPU can run: %lld\n",
           theoreticalMaxThreads);

    return 0;
}
