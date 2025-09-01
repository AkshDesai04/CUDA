/*
 * Query if the primary device has ECC (Error-Correcting Code) memory enabled and print the result.
 *
 * Approach and thought process:
 * 
 * 1. In CUDA, ECC status can be queried in two ways:
 *    - Using `cudaDeviceProp::ECCEnabled` from `cudaGetDeviceProperties`.
 *    - Using the device attribute `cudaDevAttrEccEnabled` via `cudaDeviceGetAttribute`.
 *
 * 2. The attribute method is more straightforward and is supported on devices that expose the attribute.
 *    It returns 1 if ECC is enabled, 0 if disabled, and can return an error if the attribute is not
 *    supported by the device.
 *
 * 3. The program will:
 *    - Get the current device number with `cudaGetDevice`.
 *    - Call `cudaDeviceGetAttribute` to obtain ECC status.
 *    - Check the returned status and print an appropriate message.
 *    - Handle any CUDA API errors gracefully, printing the CUDA error string.
 *
 * 4. We include basic error checking for each CUDA call.
 *
 * 5. The output will be:
 *    - "ECC is enabled on device X." if ECC is active.
 *    - "ECC is disabled on device X." if ECC is inactive.
 *    - "ECC status could not be determined for device X. Error: <error_message>" if a CUDA error occurs.
 *
 * This program is self‑contained, written in standard C for CUDA, and compiles with `nvcc`.
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                        \
    do {                                                        \
        cudaError_t err = (call);                               \
        if (err != cudaSuccess) {                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return EXIT_FAILURE;                                \
        }                                                       \
    } while (0)

int main(void)
{
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));

    int eccEnabled = 0;
    cudaError_t err = cudaDeviceGetAttribute(&eccEnabled, cudaDevAttrEccEnabled, dev);

    if (err != cudaSuccess) {
        /* The attribute may not be supported on the device. */
        fprintf(stdout,
                "ECC status could not be determined for device %d. "
                "Error: %s\n",
                dev, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (eccEnabled == 1) {
        printf("ECC is enabled on device %d.\n", dev);
    } else if (eccEnabled == 0) {
        printf("ECC is disabled on device %d.\n", dev);
    } else {
        /* Unexpected value */
        printf("ECC status returned unexpected value %d for device %d.\n",
               eccEnabled, dev);
    }

    return EXIT_SUCCESS;
}
