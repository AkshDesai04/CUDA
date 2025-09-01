/*
Aim: Write a program that attempts to set the device to an invalid index (e.g., 999) and use proper error checking to report that the device does not exist.

Thinking:
- The task requires a CUDA C (.cu) source file that demonstrates error handling when trying to set an invalid device index.
- The code should attempt to set device index 999 using cudaSetDevice.
- Since 999 is almost certainly out of range for any physical device, cudaSetDevice should return an error code.
- We'll capture the returned cudaError_t and, if it is not cudaSuccess, use cudaGetErrorString to retrieve a human‑readable error message.
- The program will print this message to stderr and exit with a non‑zero status code.
- To keep the example minimal, we only need to include the CUDA runtime header and a standard C++ header for printing.
- We won't launch any kernels; the focus is solely on device selection and error reporting.
- The code must compile with nvcc, so we avoid any advanced C++ features that require newer compiler support beyond what's typical in CUDA development environments.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

int main(void) {
    const int invalidDevice = 999;
    cudaError_t err = cudaSetDevice(invalidDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaSetDevice(%d) failed: %s\n",
                invalidDevice, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // If, improbably, the device index existed, notify success.
    printf("Successfully set device %d\n", invalidDevice);
    return EXIT_SUCCESS;
}
