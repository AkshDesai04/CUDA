```cpp
/*
Modify the function to use `cudaMemset` to initialize all bytes to `0xFF` (which corresponds to -1 for signed integers).

Thinking:
1. The goal is to replace a manual initialization of a device array with a direct call to cudaMemset.
2. cudaMemset takes a pointer, a byte value, and a size in bytes. Setting the byte value to 0xFF will
   fill every byte with 0xFF, which for a signed int results in -1.
3. Allocate an array of int on the device, then call cudaMemset(d_arr, 0xFF, N*sizeof(int)).
4. After that, copy the array back to the host to verify that all elements are -1.
5. Use a simple helper macro for error checking to keep the code readable.
6. Compile with nvcc and run; output should show -1 for each element.
*/

#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                 \
    do {                                                 \
        cudaError_t err = call;                          \
        if (err != cudaSuccess) {                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__,                  \
                    cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                          \
        }                                                \
    } while (0)

int main(void) {
    const int N = 10;
    int *d_arr = NULL;
    int *h_arr = (int *)malloc(N * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_arr, N * sizeof(int)));

    // Initialize all bytes of the device array to 0xFF using cudaMemset
    // This sets every int element to -1
    CUDA_CHECK(cudaMemset(d_arr, 0xFF, N * sizeof(int)));

    // Copy the initialized data back to host
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print the results
    printf("Device array initialized to -1 using cudaMemset:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_arr));
    free(h_arr);

    return 0;
}
```