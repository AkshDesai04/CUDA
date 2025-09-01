/*
 * Aim of the program: Query and print the device's UUID (Universally Unique ID).
 *
 * Thinking process:
 * 1. To obtain a GPU's UUID we can use the CUDA Driver API, which provides a dedicated
 *    function `cuDeviceGetUuid`.  The runtime API also exposes a UUID through
 *    `cudaDeviceProp`, but the driver API is more straightforward for this isolated
 *    query.  Therefore, we include <cuda.h> and use `cuInit`, `cuDeviceGetCount`,
 *    `cuDeviceGet`, and `cuDeviceGetUuid`.
 *
 * 2. `cuDeviceGetUuid` returns a 16‑byte `CUuuid` struct.  We need to convert this
 *    binary data into a human‑readable hexadecimal string.  A simple helper that
 *    iterates over the 16 bytes and prints each as two‑digit hex will suffice.
 *
 * 3. The program should:
 *    - Initialize the CUDA driver.
 *    - Determine how many CUDA‑capable devices are present.
 *    - For each device:
 *        * Get the device handle.
 *        * Query its UUID.
 *        * Print the device index and the UUID in hex.
 *    - Handle any errors by printing the CUDA error string.
 *
 * 4. Error checking is performed after each CUDA call using a helper macro
 *    `CHECK_CUDA` to keep the code concise.
 *
 * 5. Output format: "Device <index> UUID: <hex‑string>".
 *
 * The final code below implements these steps in a single .cu file.
 */

#include <stdio.h>
#include <cuda.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        CUresult err = (call);                                      \
        if (err != CUDA_SUCCESS) {                                  \
            const char *msg;                                       \
            cuGetErrorString(err, &msg);                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, msg);                      \
            return -1;                                             \
        }                                                           \
    } while (0)

/* Helper to print a CUuuid as a 32‑character hexadecimal string */
void print_uuid(const CUuuid *uuid) {
    for (int i = 0; i < 16; ++i) {
        printf("%02x", uuid->bytes[i]);
    }
}

int main(void) {
    /* Initialize the CUDA driver */
    CHECK_CUDA(cuInit(0));

    int deviceCount = 0;
    CHECK_CUDA(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    printf("Found %d CUDA device(s).\n", deviceCount);

    for (int devIdx = 0; devIdx < deviceCount; ++devIdx) {
        CUdevice dev;
        CHECK_CUDA(cuDeviceGet(&dev, devIdx));

        CUuuid uuid;
        CHECK_CUDA(cuDeviceGetUuid(&uuid, dev));

        printf("Device %d UUID: ", devIdx);
        print_uuid(&uuid);
        printf("\n");
    }

    return 0;
}
