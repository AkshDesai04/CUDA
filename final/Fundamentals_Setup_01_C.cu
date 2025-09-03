```cuda
/*
Write a program to explicitly set the active CUDA device to 0 using `cudaSetDevice`. Verify the call was successful by then calling `cudaGetDevice` and printing the result.

Thinking:
1. The program should be a self‑contained CUDA source file (.cu) that can be compiled with nvcc.
2. We need to include the CUDA runtime header `<cuda_runtime.h>` and standard IO headers for printing.
3. In `main`, we will:
   a. Call `cudaSetDevice(0)` and capture the return value.
   b. If the call fails, print an error message using `cudaGetErrorString` and exit.
   c. Call `cudaGetDevice(&deviceId)` to retrieve the current active device.
   d. Print the device ID and a success message.
4. We should check the return value of `cudaGetDevice` as well.
5. For simplicity, we use `printf` from `<cstdio>`. No kernel launch is required.
6. The program will return 0 on success, or a non‑zero error code on failure.
*/

#include <cstdio>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set device %d: %s\n", device, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    int currentDevice = -1;
    err = cudaGetDevice(&currentDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get current device: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Successfully set and retrieved CUDA device. Current device: %d\n", currentDevice);
    return EXIT_SUCCESS;
}
```