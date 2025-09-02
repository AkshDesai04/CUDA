/*
Aim: Check for P2P support between all pairs of GPUs on your system and print a compatibility matrix.

Thinking:
- Use the CUDA Runtime API to determine the number of GPUs present (cudaGetDeviceCount).
- For each ordered pair of GPUs (i, j) where i != j, use cudaDeviceCanAccessPeer to test whether GPU i can directly access GPU j's memory.
- The result is stored in a simple integer (1 for supported, 0 for not supported). For the diagonal entries (i == j) we print an 'X' to indicate self-access.
- The output is a readable table where rows and columns are labeled by device indices.
- Basic error checking is performed on CUDA API calls; any error causes the program to terminate with an error message.
- The program is written in standard C and is intended to be compiled with nvcc into a .cu file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA capable devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("P2P compatibility matrix (%d GPUs):\n", deviceCount);

    /* Header */
    printf("    ");
    for (int i = 0; i < deviceCount; ++i) {
        printf("%4d", i);
    }
    printf("\n");

    /* Matrix */
    for (int i = 0; i < deviceCount; ++i) {
        printf("%4d", i);
        for (int j = 0; j < deviceCount; ++j) {
            if (i == j) {
                printf("%4s", "X");
                continue;
            }

            int canAccessPeer = 0;
            cudaError_t err = cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
            if (err != cudaSuccess) {
                /* Treat any error as not supported */
                canAccessPeer = 0;
            }
            printf("%4d", canAccessPeer);
        }
        printf("\n");
    }

    return EXIT_SUCCESS;
}
