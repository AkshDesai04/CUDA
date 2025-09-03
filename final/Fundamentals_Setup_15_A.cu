```c
/*
Aim of the program:
Write a program that allocates memory on the GPU, then calls cudaDeviceReset(),
and then attempts to use or free that memory. Observe the error that occurs.

Thinking:
The CUDA runtime API provides cudaMalloc to allocate device memory and
cudaFree to free it. The function cudaDeviceReset() destroys all state
associated with the current device, including all allocations. After a
cudaDeviceReset(), any device pointer that was valid before the reset
becomes invalid. Using such a pointer (e.g., for cudaMemcpy or cudaFree)
will result in an error, typically cudaErrorInvalidDevice or cudaErrorInvalidValue.
To demonstrate this, the program will:

1. Allocate a small integer array on the GPU.
2. Initialize the array on the host and copy it to the device.
3. Call cudaDeviceReset() to reset the device.
4. Attempt to copy data back from the device to the host (using the invalid
   device pointer) and then attempt to free the device memory.
5. Check and print the error codes returned by each operation so the
   observer can see the failure.

The program will use simple error checking macros to keep the code concise.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple error checking macro */
#define CHECK_CUDA(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",    \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            return EXIT_FAILURE;                                 \
        }                                                        \
    } while (0)

int main(void)
{
    const int N = 10;
    int *d_ptr = NULL;
    int h_host[N];

    /* Initialize host data */
    for (int i = 0; i < N; ++i) h_host[i] = i;

    /* Allocate device memory */
    printf("Allocating %d integers on the device...\n", N);
    CHECK_CUDA(cudaMalloc((void**)&d_ptr, N * sizeof(int)));

    /* Copy data from host to device */
    printf("Copying data from host to device...\n");
    CHECK_CUDA(cudaMemcpy(d_ptr, h_host, N * sizeof(int), cudaMemcpyHostToDevice));

    /* Reset the device */
    printf("Resetting the device...\n");
    CHECK_CUDA(cudaDeviceReset());

    /* Attempt to copy data back from device to host using the invalid pointer */
    printf("Attempting to copy data back from device to host (should fail)...\n");
    cudaError_t err = cudaMemcpy(h_host, d_ptr, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Expected error after device reset: %s\n", cudaGetErrorString(err));
    } else {
        fprintf(stderr, "Unexpected success: data copied back.\n");
    }

    /* Attempt to free the device memory using the invalid pointer */
    printf("Attempting to free device memory (should fail)...\n");
    err = cudaFree(d_ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "Expected error when freeing after reset: %s\n", cudaGetErrorString(err));
    } else {
        fprintf(stderr, "Unexpected success: memory freed.\n");
    }

    printf("Program completed.\n");
    return EXIT_SUCCESS;
}
```