```cpp
/*
Copy a sub-section of the device array (e.g., elements 100 to 149) to the beginning of the host array. This requires pointer arithmetic on both host and device pointers.

[Thinking]
The goal is to demonstrate how to use pointer arithmetic to copy a specific segment of a device array back to the host.  We allocate a host array with at least 150 elements, initialise it, and copy it to a device array.  Then we use `cudaMemcpy` with pointers offset by 100 on the device side, and the base pointer on the host side, to copy only elements 100–149 (50 elements) into the beginning of a separate host buffer.  The code includes error checking, prints a few values for verification, and cleans up resources.
*/
#include <stdio.h>
#include <cuda_runtime.h>

#define N 200          // Size of the full array
#define SUB_START 100  // Starting index of the sub-section
#define SUB_SIZE 50    // Number of elements to copy

// Simple macro for checking CUDA errors
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

int main(void)
{
    int *h_array = (int*)malloc(N * sizeof(int));
    int *h_sub  = (int*)malloc(SUB_SIZE * sizeof(int));

    if (!h_array || !h_sub) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialise host array
    for (int i = 0; i < N; ++i) {
        h_array[i] = i;
    }

    // Device array
    int *d_array = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_array, N * sizeof(int)));

    // Copy entire host array to device
    CHECK_CUDA(cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice));

    // Copy sub-section from device to host using pointer arithmetic
    // Device pointer offset: d_array + SUB_START
    // Host pointer: h_sub (points to start of destination buffer)
    CHECK_CUDA(cudaMemcpy(h_sub,
                          d_array + SUB_START,
                          SUB_SIZE * sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Verify the copy by printing the first 10 elements of h_sub
    printf("First 10 elements of the copied sub-section:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_sub[i]);
    }
    printf("\n");

    // Clean up
    free(h_array);
    free(h_sub);
    CHECK_CUDA(cudaFree(d_array));

    return EXIT_SUCCESS;
}
```