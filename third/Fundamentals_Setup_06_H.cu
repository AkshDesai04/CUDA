/*
Aim of the program: Query and print the surface alignment requirement in bytes.

Thinking:
1. We need to retrieve the surface alignment requirement for a CUDA device.
2. The CUDA Runtime API provides cudaDeviceGetAttribute which can query device properties.
3. The specific attribute for surface alignment is cudaDevAttrSurfaceAlignment (enum value CUDA_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT).
4. We'll query this attribute for device 0 (the first device). For simplicity, we assume a single GPU or that device 0 is representative.
5. We'll check the return status of cudaDeviceGetAttribute and print an error message if it fails.
6. Finally, we print the retrieved alignment value in bytes.
7. The program will be a minimal .cu file with necessary includes, main function, error checking, and output.
8. No extra code or files will be produced, only the .cu source as requested.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void)
{
    int device = 0; // Query the first CUDA device
    int surface_alignment = 0;

    cudaError_t err = cudaDeviceGetAttribute(&surface_alignment,
                                            cudaDevAttrSurfaceAlignment,
                                            device);

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetAttribute failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    printf("Surface alignment requirement: %d bytes\n", surface_alignment);
    return 0;
}
