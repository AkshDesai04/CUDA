```c
/*
 * Aim: Query and print the device's UUID (Universally Unique ID).
 *
 * Thinking process:
 * 1. CUDA provides a struct cudaDeviceProp that contains a field of type
 *    cudaUUID_t named 'uuid'. This field holds the device's UUID as defined
 *    by the IEEE 7545 standard: a 128‑bit value split into the following
 *    components:
 *      - time_low            (uint32_t)
 *      - time_mid            (uint16_t)
 *      - time_hi_and_version (uint16_t)
 *      - clock_seq           (uint16_t)
 *      - node[6]             (uint8_t[6])
 * 2. To display the UUID in the familiar textual form
 *    "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" we format each component with
 *    hexadecimal printing.  The node part consists of six bytes that are
 *    printed sequentially as twelve hexadecimal digits.
 * 3. The program will:
 *      a) Query the number of CUDA devices using cudaGetDeviceCount.
 *      b) For each device, retrieve its cudaDeviceProp via
 *         cudaGetDeviceProperties.
 *      c) Extract the UUID components and print them in the standard format.
 *      d) Perform minimal error checking on CUDA API calls.
 *
 * The code uses standard C I/O (printf) and the CUDA runtime API. It can be
 * compiled with `nvcc uuid_query.cu -o uuid_query` and executed on a machine
 * with a CUDA‑capable GPU.
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            return EXIT_FAILURE;                                    \
        }                                                           \
    } while (0)

int main(void)
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        /* Extract UUID components */
        const cudaUUID_t *uuid = &prop.uuid;
        printf("Device %d (%s) UUID: %08x-%04x-%04x-%04x-"
               "%02x%02x%02x%02x%02x%02x\n",
               dev,
               prop.name,
               uuid->time_low,
               uuid->time_mid,
               uuid->time_hi_and_version,
               uuid->clock_seq,
               uuid->node[0], uuid->node[1], uuid->node[2],
               uuid->node[3], uuid->node[4], uuid->node[5]);
    }

    return EXIT_SUCCESS;
}
```