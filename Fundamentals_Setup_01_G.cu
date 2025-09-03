/*
Aim of the program: Query and print the total number of registers available per thread block on the primary device.

Thinking process:
1. We need to use CUDA runtime API to query device properties.
2. The relevant field in cudaDeviceProp is 'regsPerBlock', which gives the number of 32-bit registers available per thread block.
3. The program should:
   - Check that at least one CUDA device is present.
   - Select the primary device (device 0).
   - Retrieve the device properties using cudaGetDeviceProperties.
   - Print the device name and the regsPerBlock value.
4. Error handling is added for robustness: if any CUDA API call fails, an error message is printed and the program exits with non-zero status.
5. Since the request is for a CUDA C (.cu) file, we use C headers (<stdio.h>) and CUDA runtime headers (<cuda_runtime.h>).
6. The code is self‑contained and can be compiled with nvcc.

The final program follows these steps and outputs the desired information.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro for checking CUDA errors */
#define CHECK_CUDA(call)                                 \
    do {                                                 \
        cudaError_t err = (call);                        \
        if (err != cudaSuccess) {                        \
            fprintf(stderr,                              \
                    "CUDA error at %s:%d - %s (%d)\n",   \
                    __FILE__, __LINE__,                 \
                    cudaGetErrorString(err), err);       \
            return EXIT_FAILURE;                         \
        }                                                \
    } while (0)

int main(void)
{
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    int device = 0; /* primary device */
    CHECK_CUDA(cudaSetDevice(device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("Total registers available per thread block on device %d (%s): %d\n",
           device, prop.name, prop.regsPerBlock);

    return EXIT_SUCCESS;
}
