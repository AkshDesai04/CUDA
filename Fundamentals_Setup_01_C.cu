```cpp
/* 
Write a program to explicitly set the active CUDA device to 0 using `cudaSetDevice`. Verify the call was successful by then calling `cudaGetDevice` and printing the result.

Thinking process:
1. Include necessary headers: <cuda_runtime.h> for CUDA runtime API and <cstdio> for printing.
2. In main, call cudaSetDevice(0). Check the return value; if not cudaSuccess, print an error and exit.
3. Call cudaGetDevice to retrieve the currently active device. Again check the return value for errors.
4. Print the device number obtained from cudaGetDevice, which should be 0 if everything worked.
5. Return 0 from main to indicate successful execution.
6. Ensure the code is self-contained in a single .cu file so it can be compiled with nvcc.
*/

#include <cuda_runtime.h>
#include <cstdio>

int main(void) {
    int device = 0;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaSetDevice(%d) failed: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    int currentDevice = -1;
    err = cudaGetDevice(&currentDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Successfully set active device to %d. Current device is %d.\n", device, currentDevice);

    return 0;
}
```