```cuda
/*
Demonstrate catching an asynchronous error from inside the kernel (e.g. an illegal memory access) using CHECK(cudaDeviceSynchronize()).

The goal of this program is to show how CUDA reports errors that occur during kernel execution, which are asynchronous with respect to the host.  When a kernel launches, CUDA immediately returns control to the host.  Any error that occurs within the kernel (for example, an illegal memory access) will not be reported until a synchronization point such as cudaDeviceSynchronize() or cudaMemcpy() is reached.  This program deliberately causes an out‑of‑bounds write inside the kernel.  After the kernel launch, the host calls CHECK(cudaDeviceSynchronize()), which checks the returned cudaError_t value.  If an error occurred during kernel execution, the error string is printed and the program exits.

Key points covered:
1. Definition of a CHECK macro to simplify error handling.
2. Allocation of device memory of a small size.
3. A kernel that writes beyond the bounds of the allocated array (illegal memory access).
4. Launching the kernel and then synchronizing with CHECK to capture the error.
5. Printing a message only if the kernel succeeded, otherwise printing the error.

The code is fully self‑contained, compiles with `nvcc`, and demonstrates the asynchronous nature of CUDA error reporting.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro to check CUDA API calls and kernel launches
#define CHECK(call)                                          \
    do {                                                     \
        cudaError_t err = (call);                            \
        if (err != cudaSuccess) {                            \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                              \
        }                                                    \
    } while (0)

// Kernel that intentionally writes out of bounds
__global__ void illegal_access_kernel(int *d_arr, size_t size) {
    // Use a single thread to keep things simple
    if (threadIdx.x == 0) {
        // Attempt to write beyond the allocated array
        // This is an illegal memory access and will trigger an error
        d_arr[size + 10] = 42; // Out-of-bounds write
    }
}

int main(void) {
    const size_t N = 10;          // Small array size
    const size_t bytes = N * sizeof(int);

    int *d_arr = NULL;
    // Allocate device memory
    CHECK(cudaMalloc((void **)&d_arr, bytes));

    // Initialize device memory to zero
    CHECK(cudaMemset(d_arr, 0, bytes));

    // Launch the kernel with one block and one thread
    illegal_access_kernel<<<1, 1>>>(d_arr, N);

    // Synchronize and check for errors
    CHECK(cudaDeviceSynchronize());

    // If we reach this point, no error was detected
    printf("Kernel executed successfully.\n");

    // Clean up
    CHECK(cudaFree(d_arr));
    CHECK(cudaDeviceReset());
    return 0;
}
```