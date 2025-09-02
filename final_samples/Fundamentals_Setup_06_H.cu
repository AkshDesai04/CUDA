/*
Aim: Query and print the surface alignment requirement in bytes.

Thinking:
To obtain the surface alignment requirement, we use the CUDA runtime API function
cudaDeviceGetAttribute. The attribute to query is cudaDevAttrSurfaceAlignment,
which provides the minimum alignment (in bytes) required for surfaces on the
specified device. We first obtain the current device with cudaGetDevice, then
call cudaDeviceGetAttribute to retrieve the value. If the call fails we print
an error message; otherwise we print the alignment requirement.

The program is a simple single-threaded host program and does not launch any
kernels or use device memory. It includes minimal error checking and uses the
standard CUDA runtime header <cuda_runtime.h>.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(err)                                               \
    do {                                                              \
        cudaError_t _e = (err);                                       \
        if (_e != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(_e));      \
            return -1;                                               \
        }                                                             \
    } while (0)

int main(void) {
    int device = 0;
    // Get the current device
    CUDA_CHECK(cudaGetDevice(&device));

    int surfaceAlignment = 0;
    // Query the surface alignment attribute
    CUDA_CHECK(cudaDeviceGetAttribute(&surfaceAlignment,
                                      cudaDevAttrSurfaceAlignment,
                                      device));

    printf("Surface alignment requirement for device %d: %d bytes\n",
           device, surfaceAlignment);

    return 0;
}
