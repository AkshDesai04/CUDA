```cuda
/*
Write a program that queries and prints the clock rate of the primary GPU in kilohertz (kHz).

Thinking:
The CUDA runtime API provides the function `cudaGetDeviceProperties` which fills a
`cudaDeviceProp` structure with information about a GPU device. One of the fields
in this structure is `clockRate`, which reports the device's clock frequency in
kilohertz (kHz). The program should therefore:

1. Include the CUDA runtime header (`cuda_runtime.h`).
2. In `main`, call `cudaGetDeviceProperties` for device index 0 (the primary GPU).
3. Check the return status for errors.
4. Print the `clockRate` field along with a descriptive message.
5. Exit with a suitable return code.

Edge cases:
- If there are no CUDA devices, `cudaGetDeviceProperties` will return an error.
- `clockRate` is already in kHz, so no conversion is needed.

This straightforward implementation demonstrates basic CUDA device interrogation and
provides a useful diagnostic output.
*/
#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0; // primary GPU
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving device properties: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    printf("Primary GPU (device %d) clock rate: %d kHz\n", device, prop.clockRate);
    return 0;
}
```