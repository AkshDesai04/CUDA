/*
Write a program that queries and prints the clock rate of the primary GPU in kilohertz (kHz).

Thinking:
- The program must be written in CUDA C and produce a .cu file.
- The primary GPU refers to device index 0.
- CUDA runtime API provides cudaDeviceProp struct which contains clockRate in kilohertz.
- Steps:
  1. Include <stdio.h> for printf and <cuda_runtime.h> for CUDA functions.
  2. In main(), set device to 0 using cudaSetDevice(0) for safety.
  3. Declare cudaDeviceProp variable and call cudaGetDeviceProperties(&prop, 0).
  4. The prop.clockRate field is already in kHz, so print it directly.
  5. Handle any errors with simple cudaGetLastError checks or exit on failure.
- Output format: "Clock rate: <value> kHz".
- No additional output or dependencies. The code will be a complete .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0;
    cudaError_t err;

    // Optional: set the device explicitly (though device 0 is default)
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA device %d: %s\n", device, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get properties for device %d: %s\n", device, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // clockRate is reported in kHz
    printf("Clock rate: %d kHz\n", prop.clockRate);

    return EXIT_SUCCESS;
}
