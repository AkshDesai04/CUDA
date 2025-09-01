/*
Aim of program: Trigger cudaErrorInvalidValue by passing an invalid kind to cudaMemcpy,
e.g., (cudaMemcpyKind)123.

Thinking process:
1. We need a minimal CUDA C program that can be compiled with nvcc and executed on a system with a CUDA-capable GPU.
2. The goal is to deliberately cause a cudaMemcpy call to fail with the specific error cudaErrorInvalidValue.
   The API documentation states that an invalid value for the `kind` parameter results in this error.
3. The simplest way to trigger this is to cast an arbitrary integer that is not one of the defined
   cudaMemcpyKind values to cudaMemcpyKind. For example, (cudaMemcpyKind)123.
4. The program will:
   - Allocate a small array on the host.
   - Allocate a corresponding array on the device.
   - Call cudaMemcpy with the invalid kind.
   - Capture the returned cudaError_t.
   - Print the error code and human‑readable string using cudaGetErrorString.
   - Clean up resources and exit.
5. The program is intentionally straightforward: it does not use any helper macros for error checking
   beyond a single print statement. This keeps the example focused on the error generation itself.
6. All CUDA API calls are checked for errors only where necessary; the key call (cudaMemcpy with invalid
   kind) is expected to return cudaErrorInvalidValue.
7. We use cudaMallocHost for pinned host memory to avoid complications, but any allocation would work.
8. Finally, we reset the device to ensure a clean state.

This file is a complete .cu source that can be compiled with:
   nvcc -o trigger_invalid_kind trigger_invalid_kind.cu
and run with:
   ./trigger_invalid_kind
to observe the cudaErrorInvalidValue being triggered.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

int main(void)
{
    const size_t N = 10;
    int *h_a = nullptr;
    int *d_a = nullptr;

    // Allocate pinned host memory for simplicity
    cudaError_t err = cudaMallocHost((void**)&h_a, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Initialize host array
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = static_cast<int>(i);
    }

    // Allocate device memory
    err = cudaMalloc((void**)&d_a, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_a);
        return EXIT_FAILURE;
    }

    // Intentionally pass an invalid cudaMemcpyKind to trigger cudaErrorInvalidValue
    err = cudaMemcpy(d_a, h_a, N * sizeof(int), static_cast<cudaMemcpyKind>(123));

    // Print the error result
    printf("cudaMemcpy returned error code: %d\n", static_cast<int>(err));
    printf("Error message: %s\n", cudaGetErrorString(err));

    // Clean up resources
    cudaFree(d_a);
    cudaFreeHost(h_a);

    // Reset the device to ensure a clean exit
    cudaDeviceReset();

    return 0;
}
