```cpp
/*
Copy a sub-section of the device array (e.g., elements 100 to 149) to the beginning of the host array. This requires pointer arithmetic on both host and device pointers.

Thought process:
1. Allocate a host array of sufficient size (here 200 integers) and initialize it with known values.
2. Allocate a device array of the same size.
3. Copy the entire host array to the device so that the device has a complete dataset.
4. Use pointer arithmetic on the device side to point to the start of the desired sub-section:
   `int *dev_ptr = dev_array + 100;`
   This points to element 100 of the device array.
5. Use pointer arithmetic on the host side to obtain the destination pointer, which is simply the start of the host array in this case:
   `int *host_ptr = host_array;`
6. Perform a `cudaMemcpy` from `dev_ptr` to `host_ptr`, specifying the number of elements (50) to copy.
7. Verify the copy by printing the first few elements of the host array.
8. Clean up allocated memory and handle errors appropriately.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

int main(void) {
    const int N = 200;          // total size of the array
    const int SUB_START = 100;  // starting index of sub-section
    const int SUB_SIZE = 50;    // number of elements in sub-section

    // Host array allocation
    int *host_array = (int *)malloc(N * sizeof(int));
    if (!host_array) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        host_array[i] = i;
    }

    // Device array allocation
    int *dev_array;
    CHECK_CUDA(cudaMalloc((void **)&dev_array, N * sizeof(int)));

    // Copy entire host array to device
    CHECK_CUDA(cudaMemcpy(dev_array, host_array, N * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Pointer arithmetic on device: point to element 100
    int *dev_ptr = dev_array + SUB_START;

    // Pointer arithmetic on host: destination is beginning of host_array
    int *host_ptr = host_array;

    // Copy sub-section from device to host
    CHECK_CUDA(cudaMemcpy(host_ptr, dev_ptr, SUB_SIZE * sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Print first 10 elements to verify copy
    printf("First 10 elements of host array after copy:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", host_array[i]);
    }
    printf("\n");

    // Optionally print the full 50 elements to confirm
    printf("Full 50 elements of host array after copy:\n");
    for (int i = 0; i < SUB_SIZE; ++i) {
        printf("%d ", host_array[i]);
    }
    printf("\n");

    // Clean up
    free(host_array);
    CHECK_CUDA(cudaFree(dev_array));

    return EXIT_SUCCESS;
}
```