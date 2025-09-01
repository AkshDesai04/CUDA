```cpp
/* 
Aim of the program: Check if the device supports cooperative launch (`cooperativeLaunch`) and print the result.

Thinking:
1. We need to query each CUDA-capable device present on the system.
2. CUDA exposes a device attribute `cudaDevAttrCooperativeLaunch` (via `cudaDeviceGetAttribute`) which indicates whether cooperative launch is supported.
3. For each device:
   - Retrieve the attribute value (0 or 1).
   - Print a human-readable message.
4. Handle errors gracefully: check the return codes of CUDA runtime API calls and report any failures.
5. The program will be compiled as a .cu file and run on a machine with CUDA installed.
6. We'll use `printf` for simplicity and avoid additional dependencies.
7. The code uses standard CUDA runtime API and no external libraries.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("Error: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        int cooperativeLaunchSupported = 0;
        err = cudaDeviceGetAttribute(&cooperativeLaunchSupported,
                                     cudaDevAttrCooperativeLaunch,
                                     dev);
        if (err != cudaSuccess) {
            printf("Error: cudaDeviceGetAttribute failed for device %d: %s\n",
                   dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d (%s): Cooperative Launch Supported: %s\n",
               dev,
               /* retrieve device name for clarity */
               [=]() {
                   cudaDeviceProp prop;
                   cudaGetDeviceProperties(&prop, dev);
                   return prop.name;
               }(),
               cooperativeLaunchSupported ? "Yes" : "No");
    }

    return 0;
}
```