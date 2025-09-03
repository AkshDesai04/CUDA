```c
/*
Query if the primary device has ECC (Error-Correcting Code) memory enabled and print the result.

Thinking:
- The program should use the CUDA Runtime API to query the ECC status of the primary (device 0) GPU.
- The CUDA API provides the `cudaDeviceGetAttribute` function, which can retrieve a variety of device attributes.
- ECC status is exposed via the `cudaDevAttrEccEnabled` attribute; the returned value is 0 if ECC is not enabled, 1 if it is.
- Steps:
  1. Include necessary headers: `<stdio.h>` for I/O and `<cuda_runtime.h>` for CUDA functions.
  2. Get the number of CUDA devices with `cudaGetDeviceCount`. If none are present, report and exit.
  3. Optionally set the device with `cudaSetDevice(0)`. While not strictly required for querying an attribute, it is a good practice.
  4. Call `cudaDeviceGetAttribute(&eccEnabled, cudaDevAttrEccEnabled, 0)` to retrieve the ECC status.
  5. Check the return value of the CUDA call for errors and handle them by printing an error message.
  6. Print whether ECC is enabled ("Yes") or not ("No") for the primary device.
- The code will be compiled with `nvcc` and run on systems with CUDA support.
- The program is intentionally simple, focusing solely on querying and reporting ECC status.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    int device = 0; // primary device
    // Set the device (optional but good practice)
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    int eccEnabled = 0;
    err = cudaDeviceGetAttribute(&eccEnabled, cudaDevAttrEccEnabled, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetAttribute (ECC) failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Device %d ECC enabled: %s\n", device, eccEnabled ? "Yes" : "No");

    return 0;
}
```