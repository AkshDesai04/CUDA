/*
Trigger `cudaErrorInvalidValue` by passing an invalid `kind` to `cudaMemcpy`, e.g., `(cudaMemcpyKind)123`.

[Thinking]
The goal of this program is to deliberately provoke a cudaErrorInvalidValue by using an invalid cudaMemcpyKind value in a cudaMemcpy call. The program will:
1. Allocate a small host array and a device array.
2. Attempt to copy the host array to the device array using a bogus memcpy kind cast from an integer (123).
3. Capture the returned cudaError_t value.
4. Print the error code and the corresponding human-readable error string.
5. Clean up any allocated resources.
This will demonstrate how CUDA reports an invalid argument error when the copy kind is not one of the defined cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, etc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    const size_t N = 10;
    int hostArray[N];
    int *deviceArray = NULL;
    cudaError_t err;

    /* Initialize host data */
    for (size_t i = 0; i < N; ++i)
        hostArray[i] = (int)i;

    /* Allocate device memory */
    err = cudaMalloc((void**)&deviceArray, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    /* Intentionally use an invalid cudaMemcpyKind value */
    cudaMemcpyKind invalidKind = (cudaMemcpyKind)123;
    err = cudaMemcpy(deviceArray, hostArray, N * sizeof(int), invalidKind);

    if (err != cudaSuccess) {
        fprintf(stderr, "Expected error: cudaMemcpy returned %d (%s)\n",
                err, cudaGetErrorString(err));
    } else {
        fprintf(stderr, "Unexpected success: cudaMemcpy should have failed.\n");
    }

    /* Clean up */
    cudaFree(deviceArray);

    return 0;
}
