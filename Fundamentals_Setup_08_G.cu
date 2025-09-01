```c
/*
Check if the device is a multi-GPU board (`isMultiGpuBoard`).

The aim of this program is to query the current CUDA device and determine
whether it belongs to a multi‑GPU board.  CUDA exposes this information
through the `cudaDeviceGetAttribute` function and the attribute
`cudaDevAttrMultiGpuBoard`.  The program performs the following steps:

1. Obtain the current device ID using `cudaGetDevice`.
2. Query the attribute `cudaDevAttrMultiGpuBoard` for that device.
   The returned value is 1 if the device is part of a multi‑GPU board,
   otherwise 0.
3. Print a clear message indicating the result.
4. Handle any CUDA API errors by printing an error message and exiting.

The code uses the CUDA runtime API, includes proper error checking,
and is ready to compile with `nvcc`.  It is self‑contained in a single
`.cu` file and demonstrates how to use the multi‑GPU board attribute
in a straightforward manner.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Macro to check CUDA API return values */
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d - %s (%d)\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err), err);      \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void)
{
    int device = 0;
    /* Get the current device; if none set, default to 0 */
    CHECK_CUDA(cudaGetDevice(&device));

    int isMultiGpuBoard = 0;
    /* Query the attribute */
    CHECK_CUDA(cudaDeviceGetAttribute(&isMultiGpuBoard,
                                      cudaDevAttrMultiGpuBoard,
                                      device));

    if (isMultiGpuBoard) {
        printf("Device %d is part of a multi-GPU board.\n", device);
    } else {
        printf("Device %d is NOT part of a multi-GPU board.\n", device);
    }

    return 0;
}
```