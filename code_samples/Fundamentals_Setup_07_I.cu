```cu
/*
Aim of the program: Check if the device supports peer-to-peer access with other GPUs (p2pAccessSupported).

Thinking process:
1. We need to query the number of GPUs on the system with cudaGetDeviceCount.
2. For each GPU, we want to check if it can access every other GPU using cudaDeviceCanAccessPeer.
3. The program will print for each device whether it supports peer-to-peer access with any other device.
4. Use helper macro CUDA_CHECK to handle error codes gracefully.
5. The program is self-contained and does not require any external libraries beyond the CUDA runtime.
6. Since we only need to output a single .cu file, we will put everything in main() and compile with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

int main(void)
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount < 1) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Found %d CUDA device(s).\n", deviceCount);

    // For each device, check peer access to every other device
    for (int src = 0; src < deviceCount; ++src) {
        bool hasPeer = false;
        printf("\nDevice %d (%s):\n", src, "");
        char name[128];
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, src));
        printf("  Name: %s\n", prop.name);
        for (int dst = 0; dst < deviceCount; ++dst) {
            if (src == dst) continue;
            int canAccess;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess, src, dst));
            if (canAccess) {
                hasPeer = true;
                printf("  Can access peer device %d (%s)\n", dst, prop.name);
            }
        }
        if (!hasPeer) {
            printf("  Does not support peer-to-peer access with any other device.\n");
        }
    }

    return EXIT_SUCCESS;
}
```