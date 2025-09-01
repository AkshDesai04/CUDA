```cpp
/*
Aim: Run a program that has an out-of-bounds memory access inside the kernel (an asynchronous error) with and without proper post-launch error checking (cudaDeviceSynchronize). Observe that without it, the program may complete without any visible error, but potentially with incorrect results.

Thinking:
1. We want to create a CUDA kernel that writes outside the bounds of a device array.  
2. The kernel will be launched with more threads than the array size.  
3. Out-of-bounds accesses are asynchronous; the launch itself will succeed, but the error will surface only after a synchronization call such as cudaDeviceSynchronize().  
4. We'll provide a command line flag ("check" or "nocheck") to decide whether to call cudaDeviceSynchronize() after the kernel launch.  
5. The program will copy the device array back to host and print it so that we can observe any incorrect or missing data.  
6. We use a simple error-checking macro for all CUDA API calls to keep the code clean.  
7. Since the out-of-bounds write may corrupt memory that isn't part of our array, the printed results may remain unchanged or show garbage; the key is that with the "check" flag we will see an error message, while without it we won't.

Note: Running this code may produce a cudaErrorIllegalAddress (or similar) when error checking is enabled. On systems where memory protection is strong, the program might crash or produce undefined results. This is for demonstration purposes only.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Simple CUDA error checking macro
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

// Kernel that writes out of bounds
__global__ void bad_kernel(int *d_arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Intentionally write beyond the array bounds
    if (idx < N) {
        d_arr[idx] = idx;      // Normal write
    } else {
        // Out-of-bounds writes
        d_arr[idx] = idx;      // This is UB
        d_arr[idx + 1] = idx + 1; // Even more UB
    }
}

int main(int argc, char *argv[]) {
    // Decide whether to perform error checking
    bool do_check = false;
    if (argc > 1 && strcmp(argv[1], "check") == 0) {
        do_check = true;
    }

    const int N = 10;                // Size of the array
    const int THREADS = 32;          // Launch more threads than N to force OOB
    const int BLOCKS = 1;

    int *h_arr = (int *)malloc(N * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    memset(h_arr, 0, N * sizeof(int));

    int *d_arr = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_arr, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch the kernel
    bad_kernel<<<BLOCKS, THREADS>>>(d_arr, N);

    if (do_check) {
        // Synchronize and check for errors
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize() reported error after kernel launch: %s\n",
                    cudaGetErrorString(sync_err));
        } else {
            printf("cudaDeviceSynchronize() succeeded (no errors detected).\n");
        }
    } else {
        // No synchronization; proceed to copy results
        printf("No cudaDeviceSynchronize() called; skipping error check.\n");
    }

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Results in host array after kernel execution:\n");
    for (int i = 0; i < N; ++i) {
        printf("h_arr[%d] = %d\n", i, h_arr[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_arr));
    free(h_arr);

    return 0;
}
```