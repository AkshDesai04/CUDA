```c
/*
 * Write a program to determine and print the 'warp size' for the default CUDA device.
 * The warp size is a fundamental property that dictates how threads are scheduled.
 * Find this value in the device properties struct.
 *
 * Thinking:
 * 1. We need to query CUDA device properties for the default device (device 0).
 * 2. Use cudaGetDeviceProperties to fill a cudaDeviceProp struct.
 * 3. The warp size is stored in the field warpSize of that struct.
 * 4. Print it to the console.
 * 5. Include basic error handling: check return codes of CUDA API calls.
 * 6. Use standard C headers and CUDA runtime header.
 * 7. Ensure the program compiles with nvcc and runs on a system with a CUDA-capable GPU.
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Macro to check CUDA errors */
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            return 1;                                                       \
        }                                                                   \
    } while (0)

int main(void) {
    int device_id = 0;  /* Default device */
    cudaDeviceProp prop;

    /* Retrieve device properties */
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));

    /* Print warp size */
    printf("Warp size of device %d ('%s') is: %d\n",
           device_id, prop.name, prop.warpSize);

    return 0;
}
```