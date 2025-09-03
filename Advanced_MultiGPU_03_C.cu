/*
Check for P2P support between all pairs of GPUs on your system and print a compatibility matrix.
The goal of this program is to query each pair of GPUs in the system using CUDA's
runtime API to determine whether peer-to-peer (P2P) memory access is possible. For
each pair (i, j) we use cudaDeviceCanAccessPeer to check compatibility, and if
compatible we enable P2P access via cudaDeviceEnablePeerAccess. The program then
prints a matrix where each entry indicates whether P2P access is possible
between the corresponding GPU pair. The diagonal entries are set to true
since a GPU can always access its own memory. After printing, any enabled
peer access is cleaned up by disabling it. The code handles CUDA errors by
printing descriptive messages and aborting if a fatal error occurs. It also
accounts for the possibility that cudaDeviceEnablePeerAccess may return
cudaErrorPeerAccessAlreadyEnabled if the access was previously enabled.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0)
    {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    // Allocate matrix for P2P capability
    bool **p2pMatrix = (bool **)malloc(deviceCount * sizeof(bool *));
    if (!p2pMatrix)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < deviceCount; ++i)
    {
        p2pMatrix[i] = (bool *)malloc(deviceCount * sizeof(bool));
        if (!p2pMatrix[i])
        {
            fprintf(stderr, "Memory allocation failed.\n");
            // Clean up previously allocated rows
            for (int k = 0; k < i; ++k)
                free(p2pMatrix[k]);
            free(p2pMatrix);
            return EXIT_FAILURE;
        }
    }

    // Check P2P capability for each pair
    for (int i = 0; i < deviceCount; ++i)
    {
        err = cudaSetDevice(i);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Error setting device %d: %s\n", i, cudaGetErrorString(err));
            // Clean up matrix memory
            for (int k = 0; k < deviceCount; ++k) free(p2pMatrix[k]);
            free(p2pMatrix);
            return EXIT_FAILURE;
        }

        for (int j = 0; j < deviceCount; ++j)
        {
            if (i == j)
            {
                p2pMatrix[i][j] = true; // Self-access always possible
                continue;
            }

            int canAccess = 0;
            err = cudaDeviceCanAccessPeer(&canAccess, i, j);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Error checking P2P from %d to %d: %s\n", i, j, cudaGetErrorString(err));
                // Clean up matrix memory
                for (int k = 0; k < deviceCount; ++k) free(p2pMatrix[k]);
                free(p2pMatrix);
                return EXIT_FAILURE;
            }

            p2pMatrix[i][j] = (canAccess == 1);

            if (canAccess)
            {
                err = cudaDeviceEnablePeerAccess(j, 0);
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled)
                {
                    fprintf(stderr, "Error enabling P2P from %d to %d: %s\n", i, j, cudaGetErrorString(err));
                    // Continue even if enabling fails; we'll mark as false
                    p2pMatrix[i][j] = false;
                }
            }
        }
    }

    // Print header
    printf("\nP2P Compatibility Matrix:\n");
    printf("    ");
    for (int j = 0; j < deviceCount; ++j)
        printf("%3d", j);
    printf("\n");

    // Print matrix
    for (int i = 0; i < deviceCount; ++i)
    {
        printf("%3d:", i);
        for (int j = 0; j < deviceCount; ++j)
        {
            printf("  %c", p2pMatrix[i][j] ? 'Y' : 'N');
        }
        printf("\n");
    }

    // Disable peer access for cleanup
    for (int i = 0; i < deviceCount; ++i)
    {
        err = cudaSetDevice(i);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Error setting device %d for cleanup: %s\n", i, cudaGetErrorString(err));
            continue;
        }
        for (int j = 0; j < deviceCount; ++j)
        {
            if (i == j) continue;
            if (p2pMatrix[i][j])
            {
                err = cudaDeviceDisablePeerAccess(j);
                if (err != cudaSuccess && err != cudaErrorPeerAccessNotEnabled)
                {
                    fprintf(stderr, "Error disabling P2P from %d to %d: %s\n", i, j, cudaGetErrorString(err));
                }
            }
        }
    }

    // Free allocated memory
    for (int i = 0; i < deviceCount; ++i)
        free(p2pMatrix[i]);
    free(p2pMatrix);

    return EXIT_SUCCESS;
}
