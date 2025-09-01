```cuda
/*
Write a program to get the ID of the currently active CUDA device using `cudaGetDevice` and print it.

Thoughts:
- The goal is straightforward: retrieve the current active CUDA device ID and output it.
- We'll use the CUDA Runtime API function `cudaGetDevice(int *device)` which fills the integer pointer with the device ID.
- For robust code, we'll check the returned `cudaError_t` value and print any error if it occurs.
- Printing will be done via `printf`.
- Include the necessary headers: <stdio.h> for I/O and <cuda_runtime.h> for CUDA runtime API.
- The program will compile as a standard .cu file. No additional command line arguments are needed.
- The main function will simply perform the query, handle errors, and output the ID.
- We don't need to set a device or launch any kernels, just query the current device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device_id = -1;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting current device: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Current active CUDA device ID: %d\n", device_id);
    return 0;
}
```