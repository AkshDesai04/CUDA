```cpp
/*
Aim of the program: Copy only the first half of the array back from the device and verify it.

Thinking and design:

1. We want a complete CUDA C program that demonstrates copying only a portion of an array
   from device to host and then verifying that portion.
2. Steps:
   - Allocate a host array of size N (e.g., 16 or 32 for simplicity).
   - Initialize the host array with some known values (e.g., i).
   - Allocate device memory of the same size.
   - Copy the entire array to the device (cudaMemcpy HtoD).
   - Launch a kernel that modifies the array (e.g., multiply each element by 2).
   - After the kernel, copy only the first half of the device array back to the host.
     This uses cudaMemcpy with the byte count set to half*N*sizeof(int).
   - Verify that the first half of the host array now contains the modified values
     (i*2) and that the second half remains unchanged from the original initialization.
   - Print results or errors if verification fails.
3. Error handling: use a helper macro to check CUDA calls.
4. Synchronize after kernel launch before copying back.
5. Clean up: free device memory.

The program will compile with nvcc and produce an executable that prints verification results.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),    \
                    cudaGetErrorString(err));                          \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Simple kernel that doubles each element of the array
__global__ void doubleArray(int *d_arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_arr[idx] = d_arr[idx] * 2;
    }
}

int main(void) {
    const int N = 16;                         // Total number of elements
    const int HALF_N = N / 2;                 // Half the number of elements
    int h_arr[N];                             // Host array
    int *d_arr = NULL;                        // Device array pointer

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_arr, N * sizeof(int)));

    // Copy entire array to device
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel: one thread per element
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    doubleArray<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy only the first half back to host
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, HALF_N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verification
    int errors = 0;
    for (int i = 0; i < HALF_N; ++i) {
        int expected = i * 2;  // kernel doubles the original value
        if (h_arr[i] != expected) {
            printf("Mismatch at index %d: expected %d, got %d\n", i, expected, h_arr[i]);
            errors++;
        }
    }

    // Verify that second half remains unchanged from original (i)
    for (int i = HALF_N; i < N; ++i) {
        if (h_arr[i] != i) {
            printf("Second half modified at index %d: expected %d, got %d\n", i, i, h_arr[i]);
            errors++;
        }
    }

    if (errors == 0) {
        printf("Verification passed: first half correctly copied and second half unchanged.\n");
    } else {
        printf("Verification failed with %d errors.\n", errors);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_arr));
    return 0;
}
```