```c
/*
Query and print whether the device supports stream priorities.

Thought process:
1. Determine what CUDA API function provides information about stream priority support.
   The runtime API provides `cudaDeviceGetAttribute` with the enum value
   `cudaDevAttrStreamPrioritiesSupported`. This attribute returns 1 if the
   device supports stream priorities, 0 otherwise.
2. Obtain the current device ID with `cudaGetDevice`.
3. Call `cudaDeviceGetAttribute` for the attribute above.
4. Print a human‑readable message indicating support or lack thereof.
5. Add basic error checking: if any CUDA call fails, print the error string.
6. Keep the program minimal: a single `main` function that performs the query
   and prints the result. No additional functionality is required.
7. Ensure the file is a valid .cu source file that can be compiled with `nvcc`.

The code below follows these steps, includes necessary headers, and prints
the result to standard output.
*/
#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device;
    cudaError_t err;

    // Get the current device
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Query stream priority support
    int supportsStreamPriorities = 0;
    err = cudaDeviceGetAttribute(&supportsStreamPriorities,
                                 cudaDevAttrStreamPrioritiesSupported,
                                 device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetAttribute failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Device %d stream priority support: %s\n",
           device,
           supportsStreamPriorities ? "Yes" : "No");

    return 0;
}
```