/*
Aim: Iterate through all devices and print the concurrentKernels status for each one.

Thinking:
The task requires a CUDA C program that queries all available GPU devices and reports whether each device
supports concurrent kernel execution. To achieve this, we will use the CUDA Runtime API. The workflow is
as follows:

1. Use `cudaGetDeviceCount` to determine how many CUDA-capable devices are present on the system.
2. Iterate over each device index.
3. For each device, call `cudaGetDeviceProperties` to fill a `cudaDeviceProp` struct.
4. Inspect the `concurrentKernels` field of the struct; this field is a non‑zero integer if the
   device supports concurrent kernels.
5. Print the device index and the status (e.g., "enabled" or "disabled").

Additionally, we will implement simple error handling. A helper function `checkCudaError` will take a
`cudaError_t` and, if an error occurred, print a message and exit the program. This ensures that
any unexpected CUDA runtime errors are reported promptly.

The resulting program is fully self‑contained, includes necessary headers, and can be compiled
with `nvcc`. It will produce output similar to:

```
Device 0: concurrentKernels support: enabled
Device 1: concurrentKernels support: disabled
```

if multiple GPUs are present. If no CUDA devices are found, it will inform the user accordingly.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Helper function for error checking
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    checkCudaError(err, "Failed to get device count");

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        checkCudaError(err, "Failed to get device properties");

        printf("Device %d: concurrentKernels support: %s\n",
               dev,
               prop.concurrentKernels ? "enabled" : "disabled");
    }

    return EXIT_SUCCESS;
}
