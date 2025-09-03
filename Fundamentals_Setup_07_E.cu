/*
Aim: Query and print the device's UUID (Universally Unique ID).

The goal is to write a simple CUDA program that retrieves the universally unique identifier (UUID) of a GPU device and prints it to the standard output. 
Key steps considered:

1. **Include CUDA Runtime Header** – `#include <cuda_runtime.h>` provides all necessary API declarations.
2. **Error handling helper** – A macro `CUDA_CHECK` wraps CUDA runtime calls and reports errors immediately.
3. **UUID Retrieval** – The CUDA API offers `cudaDeviceGetUuid` (or `cudaDeviceGetUuid_v2` for newer versions). The UUID is a 16‑byte structure (`cudaUUID` or `cudaUUID_t`). 
4. **Printing the UUID** – Convert the 16 bytes into a 32‑character hexadecimal string for readability.
5. **Device selection** – For simplicity, query device 0. The code first checks that at least one CUDA device exists.
6. **Main flow** – Get device count, verify existence, obtain UUID, print it, and exit.

The code is written in C style with CUDA extensions and is fully self‑contained. It can be compiled with `nvcc` and run on a machine with an NVIDIA GPU.

*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* Helper macro for error checking */
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

/* Convert uuid to a printable string */
void print_uuid(const unsigned char* uuid, size_t len)
{
    for (size_t i = 0; i < len; ++i)
    {
        printf("%02x", uuid[i]);
    }
    printf("\n");
}

int main(void)
{
    /* Get number of CUDA devices */
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }

    /* Query UUID for device 0 */
    /* Using cudaDeviceGetUuid (available in CUDA 9.0+) */
    unsigned char uuid[16];  // UUID is 16 bytes
    CUDA_CHECK(cudaDeviceGetUuid(uuid, 0));

    printf("Device 0 UUID: ");
    print_uuid(uuid, 16);

    return EXIT_SUCCESS;
}
