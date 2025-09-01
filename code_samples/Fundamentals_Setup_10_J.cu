```cuda
/*
Aim of the program:
Query and print the maximum texture dimension for a 1D texture using cudaDeviceGetAttribute with cudaDevAttrMaxTexture1DWidth.

Thinking behind this program:
- The program's purpose is to demonstrate how to use the CUDA Runtime API function
  cudaDeviceGetAttribute to retrieve device-specific attributes. Specifically, we
  want the maximum width supported for a 1D texture (cudaDevAttrMaxTexture1DWidth).
- Since the attribute is device-dependent, we must choose a device ID. For simplicity,
  we will use device 0, assuming it exists. In a production setting, one could
  enumerate devices with cudaGetDeviceCount() and query each device.
- The call signature is:
    cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device);
  The value pointer will receive the attribute's integer value.
- Error handling: We check the returned cudaError_t. If it's not cudaSuccess,
  we print an error message and exit with failure.
- Printing: We use standard C I/O (printf) to output the maximum texture width
  to the console. The value is an integer, so we use %d.
- The program is written in C and compiled with nvcc as a .cu file. No device
  kernel launch is required; everything is done on the host.
- Edge cases: If device 0 is not present or the attribute query fails, the
  program will report the error. Otherwise, it will print the maximum width.
- We include <stdio.h> for printf and <cuda_runtime.h> for CUDA API.
- The program is self-contained and compiles with a typical CUDA toolkit install.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0; // Device ID to query
    int maxTex1DWidth = 0;

    cudaError_t err = cudaDeviceGetAttribute(&maxTex1DWidth,
                                             cudaDevAttrMaxTexture1DWidth,
                                             device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetAttribute failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    printf("Maximum 1D texture width on device %d: %d\n", device, maxTex1DWidth);
    return 0;
}
```