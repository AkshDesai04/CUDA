/*
Aim: Write a program that deliberately triggers a `cudaErrorInvalidValue` by passing a `nullptr` as the destination for `cudaMemcpy`. Use your `CHECK()` macro to catch and report the error cleanly.

Thinking:
- The program will allocate a small buffer on the device to satisfy the `cudaMemcpy` size requirement.
- It will then attempt to copy data from the host to a `nullptr` destination using `cudaMemcpy`. 
- Passing `nullptr` as the destination pointer is invalid and should produce `cudaErrorInvalidValue`.
- A `CHECK()` macro will wrap each CUDA API call, checking the return code, printing an informative message if an error occurs, and terminating the program.
- After triggering the error, the program won't reach the cleanup steps because `CHECK` will exit. However, the cleanup code is provided for completeness in case the macro were disabled or the error were ignored.
- The program uses `cudaDeviceReset()` at the end to clean up the device context before exiting.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// CHECK macro: verifies CUDA error status and reports if an error occurs
#define CHECK(call)                                                     \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

int main(void)
{
    // Allocate a small device buffer just to satisfy size parameter of cudaMemcpy
    int *devPtr = nullptr;
    CHECK(cudaMalloc(&devPtr, sizeof(int)));

    // Host data to copy (value doesn't matter)
    int hostVal = 1234;

    // Deliberately pass nullptr as destination to trigger cudaErrorInvalidValue
    // This call is expected to fail and be caught by CHECK
    CHECK(cudaMemcpy(nullptr, &hostVal, sizeof(int), cudaMemcpyHostToDevice));

    // The following cleanup code won't be executed because CHECK will exit on error
    CHECK(cudaFree(devPtr));
    CHECK(cudaDeviceReset());

    return 0;
}
