```cuda
/*
Aim: Trigger cudaErrorInvalidValue by calling cudaSetDevice with a negative device number.

Thinking:
The task is to create a minimal CUDA program that demonstrates the generation of the specific error
`cudaErrorInvalidValue`. According to the CUDA Runtime API, passing an invalid device index to
`cudaSetDevice()` should return this error code. The most straightforward invalid index is a
negative integer, such as -1. The program will:
1. Attempt to set the device to -1.
2. Capture the returned error code.
3. Print the error code and a human-readable description using `cudaGetErrorString`.
4. Exit with a non-zero status if the error occurs, otherwise print success.

To keep the code self-contained and focused, we won't launch any kernels or perform any GPU work.
We will include the necessary headers (`cuda_runtime.h`), use `cudaSetDevice` directly, and check
the result. The program should compile with `nvcc` and, when executed, will output the error
information indicating that the call failed due to an invalid device index.

This satisfies the requirement to trigger and report `cudaErrorInvalidValue`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int negative_device = -1;
    cudaError_t err = cudaSetDevice(negative_device);

    if (err != cudaSuccess) {
        printf("cudaSetDevice(%d) failed with error: %s (code %d)\n",
               negative_device,
               cudaGetErrorString(err),
               err);
        return 1;  // Indicate failure
    } else {
        printf("cudaSetDevice(%d) succeeded unexpectedly.\n", negative_device);
        return 0;  // Indicate success (should not happen)
    }
}
```