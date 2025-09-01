/*
Query and print the maximum number of threads per multiprocessor for the default device.

Thinking:
- The goal is to retrieve the maximum number of threads that can be scheduled on a single multiprocessor
  for the currently selected CUDA device.
- We first get the ID of the current device with cudaGetDevice().
- Then we obtain the device properties via cudaGetDeviceProperties(), which fills a cudaDeviceProp structure.
- The field maxThreadsPerMultiProcessor in this structure holds the desired value.
- We'll print this integer to standard output.
- Basic error checking is included: if any CUDA call fails, the program prints the error string and exits.
- The program uses the CUDA runtime API, so it compiles with nvcc.
- This is a minimal, self-contained .cu file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void) {
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("Maximum number of threads per multiprocessor on device %d (%s): %d\n",
           device, prop.name, prop.maxThreadsPerMultiProcessor);

    return 0;
}
