```cpp
/*
Write a `CHECK_KERNEL_LAUNCH()` macro that calls `cudaPeekAtLastError()` and `cudaDeviceSynchronize()` and reports any errors.

Thinking:
- The macro should capture the last error from the kernel launch via `cudaPeekAtLastError()`.
- If an error occurred, it should print a meaningful message using `cudaGetErrorString()` and terminate the program.
- After the launch, we must synchronize the device with `cudaDeviceSynchronize()` to ensure the kernel has finished executing.
- If the synchronization fails, we again report the error and exit.
- The macro will be wrapped in a `do { ... } while(0)` block to make it safe to use with a trailing semicolon and to avoid multiple evaluation issues.
- We'll include the necessary headers: `<stdio.h>`, `<stdlib.h>`, and `<cuda_runtime.h>`.
- For demonstration, a simple kernel and a `main()` function will be provided, but the core requirement is the macro definition.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro to check the result of a kernel launch and device synchronization */
#define CHECK_KERNEL_LAUNCH()                                         \
    do {                                                             \
        cudaError_t err = cudaPeekAtLastError();                     \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "Kernel launch error: %s\n",              \
                    cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
        err = cudaDeviceSynchronize();                               \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "Kernel launch synchronization error: %s\n", \
                    cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

/* Simple kernel that does nothing but can be used to test the macro */
__global__ void dummy_kernel() {
    // No operation
}

int main() {
    // Launch the dummy kernel with 1 block of 1 thread
    dummy_kernel<<<1, 1>>>();
    CHECK_KERNEL_LAUNCH(); // Use the macro to verify launch and sync

    printf("Kernel executed successfully.\n");
    return 0;
}
```