```c
/*
Aim: Query and print the device's multiprocessor count.
 
The program uses the CUDA Runtime API to retrieve the number of streaming multiprocessors (SMs) on the selected GPU device. 
First, it queries the number of CUDA-capable devices available. Then, it selects device 0 by default (the first device). 
Using cudaGetDeviceProperties, it obtains a cudaDeviceProp structure that contains many properties of the device, including smCount, which represents the number of multiprocessors. 
The program prints this count to standard output. 
Error checking is performed after each CUDA API call to ensure the program fails gracefully if any step fails. 
The code is written in standard C with CUDA extensions and can be compiled with nvcc as follows:
    nvcc -o multiprocessor_count multiprocessor_count.cu
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro to check CUDA API call results
#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = (call);                                  \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return EXIT_FAILURE;                                  \
        }                                                          \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable device found.\n");
        return EXIT_FAILURE;
    }

    int device = 0; // Use the first device
    CHECK_CUDA(cudaSetDevice(device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("Device %d (%s) has %d multiprocessor(s).\n",
           device, prop.name, prop.multiProcessorCount);

    return EXIT_SUCCESS;
}
```