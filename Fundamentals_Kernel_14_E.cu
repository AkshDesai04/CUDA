/*
Aim: Write a `CHECK_KERNEL_LAUNCH()` macro that calls `cudaPeekAtLastError()` and `cudaDeviceSynchronize()` and reports any errors.

Thinking:
The requirement is to provide a macro that will check for any CUDA errors immediately after a kernel launch. In CUDA, the common pattern is to call `cudaPeekAtLastError()` right after launching a kernel to capture any asynchronous launch error, and then call `cudaDeviceSynchronize()` to force the host to wait for the kernel to finish, allowing any runtime errors that occur during execution to surface. The macro should report any errors found by printing them to `stderr`. I will include the standard CUDA runtime header and stdio for printing. The macro will capture the error code from both calls, check if it's not `cudaSuccess`, and print the error string using `cudaGetErrorString`. I will also print the line and file where the macro is invoked to help with debugging. No return or exit logic is required by the prompt, just reporting the error. The macro will be defined in a `.cu` file so that it can be used in any CUDA C/C++ program.

Implementation:
- Include `<cuda_runtime.h>` and `<stdio.h>`.
- Define `CHECK_KERNEL_LAUNCH()` macro.
- Inside the macro, call `cudaPeekAtLastError()` and store the result.
- If the result is not `cudaSuccess`, print the error with `cudaGetErrorString`.
- Call `cudaDeviceSynchronize()` and check the result similarly.
- Print the file and line where the macro was called using `__FILE__` and `__LINE__`.

The macro is self-contained and can be used after any kernel launch.
*/

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_KERNEL_LAUNCH()                                 \
    do {                                                      \
        cudaError_t err = cudaPeekAtLastError();              \
        if (err != cudaSuccess) {                            \
            fprintf(stderr,                                   \
                    "[CUDA Error] %s at %s:%d\n",             \
                    cudaGetErrorString(err),                  \
                    __FILE__, __LINE__);                     \
        }                                                     \
        err = cudaDeviceSynchronize();                        \
        if (err != cudaSuccess) {                            \
            fprintf(stderr,                                   \
                    "[CUDA Error] %s at %s:%d\n",             \
                    cudaGetErrorString(err),                  \
                    __FILE__, __LINE__);                     \
        }                                                     \
    } while (0)

/* Example usage (optional):
__global__ void kernel(void) { /* kernel code * / }

int main(void) {
    kernel<<<1,1>>>();
    CHECK_KERNEL_LAUNCH();
    return 0;
}
*/
