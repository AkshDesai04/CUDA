/*
 * Aim: Query and print whether the device supports stream priorities.
 * 
 * Thinking:
 * 1. CUDA provides a device attribute `cudaDevAttrStreamPrioritiesSupported` that can be queried via
 *    `cudaDeviceGetAttribute`. The attribute value is non‑zero (typically 1) if the device
 *    supports stream priorities, otherwise 0. If the attribute is not supported by the
 *    runtime, `cudaDeviceGetAttribute` will return an error (e.g. `cudaErrorNotSupported`).
 *
 * 2. The program will:
 *    - Retrieve the current device ID with `cudaGetDevice`.
 *    - Call `cudaDeviceGetAttribute` for the attribute `cudaDevAttrStreamPrioritiesSupported`.
 *    - Check the returned error code. If it is not `cudaSuccess`, report that the query failed
 *      (this can happen on very old CUDA runtime versions that do not expose this attribute).
 *    - If the query succeeded, print "Device supports stream priorities: Yes" or "No" based on
 *      the attribute value.
 *
 * 3. For robust error handling, a simple macro `CHECK_CUDA` will be used to check CUDA
 *    function return values and abort with a meaningful message if an error occurs.
 *
 * 4. The code is written in plain C++ (compatible with CUDA C) and can be compiled with
 *    `nvcc` as a `.cu` file. It includes only the necessary headers: `<stdio.h>` and
 *    `<cuda_runtime.h>`.
 *
 * 5. The program is self‑contained; no external libraries or helper files are required.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));      \
            return EXIT_FAILURE;                                             \
        }                                                                     \
    } while (0)

int main(void)
{
    int device = 0;
    // Get the current device
    CHECK_CUDA(cudaGetDevice(&device));

    int supports = 0;
    cudaError_t err = cudaDeviceGetAttribute(&supports,
                                            cudaDevAttrStreamPrioritiesSupported,
                                            device);

    if (err != cudaSuccess) {
        // If the attribute is not supported by the runtime, we consider it as unsupported
        fprintf(stderr,
                "Failed to query stream priority support (error: %s). "
                "Assuming device does not support stream priorities.\n",
                cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Device %d supports stream priorities: %s\n",
           device,
           supports ? "Yes" : "No");

    return EXIT_SUCCESS;
}
