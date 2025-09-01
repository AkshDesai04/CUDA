/*
Aim: Convert and print the clock rate in gigahertz (GHz).

Thinking:
1. The CUDA runtime API provides the function `cudaGetDeviceProperties` which fills a `cudaDeviceProp` struct.
2. The `cudaDeviceProp` struct contains a member `clockRate` measured in kilohertz (kHz).
3. To convert this to gigahertz (GHz) we divide by 1,000,000.0 (since 1 GHz = 1,000,000 kHz).
4. The program will:
   - Initialize the CUDA runtime (implicitly via querying device properties).
   - Retrieve properties for device 0.
   - Convert the clock rate to GHz.
   - Print the result to standard output.
5. Error handling: Use a macro to check return values of CUDA API calls and print descriptive messages if something goes wrong.
6. The program is written in C, compiled with nvcc, and produces a .cu file that can be built and executed on any system with CUDA support.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                           \
    do {                                                            \
        cudaError_t err = (call);                                  \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            return EXIT_FAILURE;                                   \
        }                                                           \
    } while (0)

int main(void) {
    int device_id = 0; // use the first device
    cudaDeviceProp prop;

    // Retrieve device properties
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));

    // clockRate is in kHz; convert to GHz
    double clockRateGHz = prop.clockRate / 1.0e6;

    printf("Device %d: Clock Rate = %.3f GHz\n", device_id, clockRateGHz);

    return EXIT_SUCCESS;
}
