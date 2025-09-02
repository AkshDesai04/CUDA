/*
Aim of the program: Query if the primary device has ECC (Error-Correcting Code) memory enabled and print the result.

Thinking process:
1. The program should be a CUDA C (.cu) file that uses the CUDA Runtime API to query the ECC status of the primary GPU device.
2. In CUDA Runtime, there is an attribute enum `cudaDevAttrEccEnabled` that indicates whether ECC is enabled on a device.
3. Use `cudaDeviceGetAttribute` to query this attribute for device 0 (the primary device).
4. Handle any CUDA errors gracefully by printing an error message and returning a non‑zero exit code.
5. Finally, print a human‑readable message indicating whether ECC is enabled or disabled.
6. Include necessary headers (`<stdio.h>` and `<cuda_runtime.h>`).
7. Keep the code simple and self‑contained; no external dependencies other than the CUDA Runtime library.
8. The program should compile with `nvcc` and run on a system with a CUDA capable GPU.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;  // primary device
    int eccEnabled = 0;
    cudaError_t err = cudaDeviceGetAttribute(&eccEnabled, cudaDevAttrEccEnabled, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error querying ECC status: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Device %d ECC status: %s\n", device, eccEnabled ? "Enabled" : "Disabled");
    return 0;
}
