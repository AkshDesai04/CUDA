/*
Aim of the program:
Write to a negative index, like `d_ptr[-1]`, and check the report.

Thinking:
- The goal is to deliberately trigger an invalid memory write on the GPU by accessing a negative index of a device pointer.
- We will allocate a small device array and launch a kernel with a single thread that performs the write `d_ptr[-1] = 42;`.
- Since this address is out of bounds, the CUDA runtime (or a memory checker like cuda-memcheck) should report an error.
- After launching the kernel, we call `cudaDeviceSynchronize()` to force the kernel to complete and propagate any errors.
- We then retrieve the last CUDA error using `cudaGetLastError()` and print it. If the write is caught, this should report an invalid device memory address or similar error.
- The program includes a simple error-checking macro for convenience and prints whether the kernel executed successfully or if an error occurred.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Simple macro to check CUDA API calls
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that writes to a negative index
__global__ void write_negative_index(int *d_ptr) {
    // Intentionally write to an invalid negative index
    d_ptr[-1] = 42;
}

int main(void) {
    const size_t N = 10;               // Size of device array
    int *d_ptr = nullptr;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_ptr, N * sizeof(int)));

    // Optional: initialize memory to zero
    CHECK_CUDA(cudaMemset(d_ptr, 0, N * sizeof(int)));

    // Launch kernel with one block of one thread
    write_negative_index<<<1, 1>>>(d_ptr);

    // Synchronize to ensure kernel completion and propagate errors
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize error: %s\n",
                cudaGetErrorString(sync_err));
    }

    // Check for any kernel launch errors
    cudaError_t last_err = cudaGetLastError();
    if (last_err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n",
                cudaGetErrorString(last_err));
    } else {
        printf("Kernel launched successfully (no launch error reported).\n");
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_ptr));

    // Final status
    if (sync_err != cudaSuccess || last_err != cudaSuccess) {
        printf("Negative index write caused an error as expected.\n");
    } else {
        printf("No error reported for negative index write (unexpected).\n");
    }

    return 0;
}
