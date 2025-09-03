```cuda
/*
Aim of the program:
Check if the device supports cooperative launch (`cooperativeLaunch`) and print the result.

Thinking:
1. Cooperative launch support is queried via the CUDA Runtime API attribute `cudaDevAttrCooperativeLaunch`.
2. The program will:
   - Include <cuda_runtime.h> and <stdio.h>.
   - Retrieve the number of CUDA devices using `cudaGetDeviceCount`.
   - Select the first device (device 0) for simplicity.
   - Use `cudaDeviceGetAttribute` with the attribute `cudaDevAttrCooperativeLaunch` to determine support.
   - Handle errors from CUDA API calls and print human‑readable output: "Yes" if supported, "No" otherwise.
3. The program will be written in plain C syntax and compiled with `nvcc` as a `.cu` file.
4. No external libraries are required; all functionality is provided by the CUDA Runtime.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    // Use the first device (device 0)
    int device = 0;
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error setting device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    int cooperativeLaunchSupported = 0;
    err = cudaDeviceGetAttribute(&cooperativeLaunchSupported,
                                 cudaDevAttrCooperativeLaunch,
                                 device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error querying cooperative launch attribute: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Device %d supports cooperative launch: %s\n",
           device,
           cooperativeLaunchSupported ? "Yes" : "No");

    return 0;
}
```