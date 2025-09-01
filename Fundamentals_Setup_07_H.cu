```cpp
/*
Aim of the program: Check if the device supports cooperative launch (`cooperativeLaunch`) and print the result.

Thinking process:
1. Identify the CUDA API to query device attributes. `cudaDeviceGetAttribute` is used for this purpose.
2. Find the specific attribute that indicates cooperative launch support. The attribute name is `cudaDevAttrCooperativeLaunch`. It returns 1 if the device supports cooperative kernel launches and 0 otherwise.
3. Choose a device to query. For simplicity, device 0 will be used, and we will ensure it is set as the current device.
4. Handle possible errors from CUDA API calls. Use `cudaGetLastError` and `cudaGetErrorString` for diagnostics.
5. Print the result in a user-friendly manner: "Cooperative launch support: Yes" or "Cooperative launch support: No".
6. Compile as a .cu file, which requires inclusion of `<cuda_runtime.h>` and appropriate `#include <stdio.h>`.
7. The program is self-contained and does not launch any kernels; it merely reports the capability.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceId = 0;
    int supportsCooperativeLaunch = 0;
    cudaError_t err;

    // Set the device (in case another device is currently active)
    err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Query cooperative launch support
    err = cudaDeviceGetAttribute(&supportsCooperativeLaunch,
                                 cudaDevAttrCooperativeLaunch,
                                 deviceId);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetAttribute failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print result
    printf("Cooperative launch support: %s\n",
           supportsCooperativeLaunch ? "Yes" : "No");

    return 0;
}
```