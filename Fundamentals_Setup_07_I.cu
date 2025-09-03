```c
/* 
Aim of the program:
Check if the device supports peer-to-peer access with other GPUs (`p2pAccessSupported`). 
This means for each pair of GPUs in the system we will query the CUDA runtime API
`cudaDeviceCanAccessPeer()` to determine whether one device can directly read from
another. The program enumerates all devices, and for each unique pair prints a
message indicating whether P2P is supported. If only a single GPU is present
the program will note that there are no peers to check.

Thinking:
1. Use `cudaGetDeviceCount()` to get the number of GPUs.
2. If less than 2, print a message that no peers exist.
3. Otherwise, loop over all device indices `i` and `j` where `j > i` to avoid
   duplicate pairs.
4. For each pair, call `cudaDeviceCanAccessPeer(i, j)`. This returns an
   integer: 1 if peer access is possible, 0 otherwise. It also may return an
   error if the call itself fails; check the returned `cudaError_t` and print
   a helpful message.
5. Compile with `nvcc` and run; output will be printed to stdout.

This program is minimal, self-contained, and uses only the CUDA runtime API,
so it will compile on any system with the CUDA toolkit installed.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Number of CUDA devices detected: %d\n", deviceCount);

    if (deviceCount < 2) {
        printf("Not enough devices to check peer-to-peer access.\n");
        return 0;
    }

    // Iterate over all unique device pairs
    for (int src = 0; src < deviceCount; ++src) {
        for (int dst = src + 1; dst < deviceCount; ++dst) {
            int canAccess = 0;
            err = cudaDeviceCanAccessPeer(&canAccess, src, dst);
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "Error checking p2p from device %d to device %d: %s\n",
                        src, dst, cudaGetErrorString(err));
                continue;
            }

            printf("Device %d can%s access Device %d directly.\n",
                   src,
                   canAccess ? "" : " NOT",
                   dst);

            // Also check the reverse direction
            err = cudaDeviceCanAccessPeer(&canAccess, dst, src);
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "Error checking p2p from device %d to device %d: %s\n",
                        dst, src, cudaGetErrorString(err));
                continue;
            }

            printf("Device %d can%s access Device %d directly.\n",
                   dst,
                   canAccess ? "" : " NOT",
                   src);
        }
    }

    return 0;
}
```