```cpp
/* 
Aim: 
Write a program that deliberately triggers a `cudaErrorInvalidDevice` by trying to `cudaSetDevice()` to a device index that doesn't exist (e.g., 99). Use your `CHECK()` macro to catch it. 

Thinking:
To achieve this we need a CUDA C program that attempts to set the device to an index that is guaranteed not to exist. 
The safest way is to use an index that is far beyond the number of available GPUs, e.g., 99. 
We'll define a `CHECK` macro that wraps any CUDA API call, checks the returned `cudaError_t`, and if it is not `cudaSuccess`, prints an error message and exits. 
The main function will:
1. Print the number of available devices (optional, for user context).
2. Call `CHECK(cudaSetDevice(99));` which will trigger a `cudaErrorInvalidDevice`.
Because `CHECK` handles the error, the program will exit after printing the error string. 
The code will include necessary headers, compile as a `.cu` file, and be self‑contained. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro to check CUDA calls */
#define CHECK(call)                                                     \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in file %s at line %d: %s\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK(cudaGetDeviceCount(&deviceCount));
    printf("Number of CUDA devices available: %d\n", deviceCount);

    printf("Attempting to set device to an invalid index (99)...\n");
    /* This will deliberately fail with cudaErrorInvalidDevice */
    CHECK(cudaSetDevice(99));

    /* If we reach this point, something unexpected happened */
    printf("Unexpectedly succeeded in setting device to 99.\n");
    return 0;
}
```