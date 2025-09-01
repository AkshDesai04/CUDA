/*
Check if the device supports peer-to-peer access with other GPUs (`p2pAccessSupported`).

My thinking:
To determine whether a CUDA-enabled device can access memory on another device directly (peer-to-peer), we can use the CUDA Runtime API function `cudaDeviceCanAccessPeer`. This function reports whether the CUDA driver supports peer access between two devices. The program will:
1. Query the total number of CUDA devices present.
2. Iterate over each pair of distinct devices (i, j).
3. For each pair, call `cudaDeviceCanAccessPeer(i, j)` to see if device `i` can access device `j`.
4. Output the result for each pair.
5. Optionally, if peer access is supported, we can enable it using `cudaDeviceEnablePeerAccess`, but for the purpose of this check we will just report support.
The program will include basic error handling via a macro that checks CUDA return codes and prints any errors. It can be compiled with `nvcc` and run on a system with multiple GPUs.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                  \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount < 1) {
        printf("No CUDA-capable device is detected.\n");
        return EXIT_SUCCESS;
    }

    printf("Number of CUDA devices: %d\n\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        for (int j = 0; j < deviceCount; ++j) {
            if (i == j) {
                continue; // Skip self pair
            }
            int canAccessPeer = 0;
            CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccessPeer, i, j));
            if (canAccessPeer) {
                printf("Device %d CAN access Device %d (peer-to-peer supported).\n", i, j);
                /* Optionally enable peer access:
                int enablePeer = 0;
                CHECK_CUDA(cudaDeviceEnablePeerAccess(j, 0));
                */
            } else {
                printf("Device %d CANNOT access Device %d (peer-to-peer NOT supported).\n", i, j);
            }
        }
    }

    return EXIT_SUCCESS;
}
