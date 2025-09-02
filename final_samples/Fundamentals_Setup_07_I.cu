/*
Aim: Check if the device supports peer-to-peer access with other GPUs (`p2pAccessSupported`).

Thinking:
1. The CUDA Runtime API does not expose a function named `p2pAccessSupported`.  
   The functionality to determine if two GPUs can access each other’s memory is
   provided by `cudaDeviceCanAccessPeer()`.  
2. The program should enumerate all CUDA capable devices on the system using
   `cudaGetDeviceCount()`.  
3. For each unordered pair of distinct devices (i, j) we will query
   `cudaDeviceCanAccessPeer(i, j)` and `cudaDeviceCanAccessPeer(j, i)` – the
   result is typically symmetric, but we report both directions for completeness.  
4. The program will print a table indicating whether peer‑to‑peer access is
   supported between each pair of GPUs.  
5. Error handling is performed with a simple macro that aborts the program if a
   CUDA call fails, printing the file, line and error string.  
6. The code is a self‑contained .cu file that can be compiled with `nvcc` and
   executed to display the peer‑to‑peer support matrix for the system.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Helper macro to check CUDA API return status */
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                      \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);
    printf("\nPeer-to-Peer Access Support Matrix:\n\n");

    /* Header row */
    printf("%10s", "");
    for (int j = 0; j < deviceCount; ++j) {
        printf("  %2d", j);
    }
    printf("\n");

    /* For each device i, query each device j */
    for (int i = 0; i < deviceCount; ++i) {
        printf("%10s", "Device");
        printf("%4d", i);
        for (int j = 0; j < deviceCount; ++j) {
            if (i == j) {
                printf("   N/A");  // Same device, no peer-to-peer
                continue;
            }
            int canAccessPeer = 0;
            CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccessPeer, i, j));
            printf("   %c", canAccessPeer ? 'Y' : 'N');
        }
        printf("\n");
    }

    /* Additionally, show the bidirectional check results */
    printf("\nBidirectional Peer-to-Peer Support (Y if both directions supported):\n\n");
    printf("%10s", "");
    for (int j = 0; j < deviceCount; ++j) {
        printf("  %2d", j);
    }
    printf("\n");

    for (int i = 0; i < deviceCount; ++i) {
        printf("%10s", "Device");
        printf("%4d", i);
        for (int j = 0; j < deviceCount; ++j) {
            if (i == j) {
                printf("   N/A");
                continue;
            }
            int canAccessPeerIJ = 0, canAccessPeerJI = 0;
            CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccessPeerIJ, i, j));
            CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccessPeerJI, j, i));
            printf("   %c", (canAccessPeerIJ && canAccessPeerJI) ? 'Y' : 'N');
        }
        printf("\n");
    }

    return EXIT_SUCCESS;
}
