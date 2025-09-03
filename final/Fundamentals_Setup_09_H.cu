```cu
/*
Aim of the program: Check if the device supports local L1 cache (`localL1CacheSupported`).

Thinking Process:
1. The goal is to query the CUDA device attribute `cudaDevAttrLocalL1CacheSupported` which indicates
   whether the device supports a local L1 cache. This attribute returns an integer (0 or 1).
2. We will write a simple host program that:
   - Includes <cuda_runtime.h> to access CUDA runtime API.
   - Uses `cudaDeviceGetAttribute` to query the attribute for a given device (device 0 by default).
   - Checks the return status of the API call and prints an error message if it fails.
   - Prints the support status as a human-readable string ("Yes" or "No").
3. The program will be compiled with `nvcc` and run on the host. No device kernel code is required.
4. Since we only need to demonstrate the attribute query, we keep the code minimal and self‑contained.
5. We will place the aim and the reasoning inside a multi‑line comment at the top of the file as requested.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0;  // Default to device 0
    int localL1CacheSupported = 0;

    // Query the device attribute for local L1 cache support
    cudaError_t err = cudaDeviceGetAttribute(
        &localL1CacheSupported,
        cudaDevAttrLocalL1CacheSupported,
        device
    );

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetAttribute failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    printf("Device %d local L1 cache supported: %s\n",
           device,
           localL1CacheSupported ? "Yes" : "No");

    return 0;
}
```