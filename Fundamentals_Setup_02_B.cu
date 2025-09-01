/*
Iterate through all devices and, for each one, print its multiprocessor count.

Thinking:
1. Include the necessary headers: stdio.h for printing and cuda_runtime.h for CUDA runtime API.
2. Define a macro CUDA_CHECK to wrap CUDA API calls and report errors with the call name, error string, and line number.
3. In main():
   a. Call cudaGetDeviceCount(&deviceCount) to find how many CUDA-capable devices are present.
   b. Check for errors using CUDA_CHECK.
   c. If deviceCount == 0, print a message and exit.
   d. Loop over each device index i from 0 to deviceCount-1:
      - Declare a cudaDeviceProp struct.
      - Call cudaGetDeviceProperties(&prop, i).
      - Use CUDA_CHECK to catch errors.
      - Print the device index, name, and the number of multiprocessors via prop.multiProcessorCount.
4. Return 0 from main.
5. The program does not launch any kernels; it only queries device properties.
6. The code is written in C with CUDA extensions and should be compiled with nvcc (e.g., nvcc -o device_multiprocessors device_multiprocessors.cu).
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Found %d CUDA device(s).\n", deviceCount);
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("Device %d: %s\n", i, prop.name);
        printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);
    }

    return EXIT_SUCCESS;
}
