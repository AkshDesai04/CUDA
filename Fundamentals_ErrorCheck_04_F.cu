/*
Look up the documentation for `cudaErrorInvalidValue` and find another API call that can trigger it.

Thinking:
The `cudaErrorInvalidValue` is generated when an API receives an invalid argument.  
To demonstrate this error in a simple CUDA program, I chose the `cudaGetDeviceProperties`
function, which expects a non‑NULL pointer for the device properties structure.  
Passing a NULL pointer (and even an invalid device index such as 0 when no devices are
present) will trigger `cudaErrorInvalidValue`.  
The program calls this function with a NULL pointer, checks the returned error code,
prints the error name and description, and exits. This illustrates another API that can
produce the same error as described in the documentation. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    cudaError_t err;

    /* Intentionally trigger cudaErrorInvalidValue by passing a NULL pointer */
    err = cudaGetDeviceProperties(NULL, 0);

    if (err != cudaSuccess) {
        printf("Error triggered: %s - %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
    } else {
        printf("Unexpected success.\n");
    }

    return 0;
}
